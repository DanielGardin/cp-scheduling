from typing import Any
from collections.abc import Iterable

from cpscheduler.utils.list_utils import convert_to_list
from cpscheduler.utils.general_algo import binary_search

from cpscheduler.environment.constants import Time, Float, MAX_TIME
from cpscheduler.environment.state.events import DomainEvent
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint


class ResourceConstraint(Constraint):
    """
    Resource constraint for the scheduling environment.

    This constraint defines the renewable resources available for tasks and their usage.
    The resources can be defined as a list of capacities and a list of resource usage for each task.

    Arguments:
        capacities: Iterable[float]
            A list of capacities for each renewable resource. The length of the list should be equal to the
            number of resources.

        resource_usage: Iterable[str]
            A list of strings that define the resource usage for each task.
            Each string refers to a column in the tasks data that contains the
            resource usage for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    resource_tags: list[str]
    capacities: list[float]
    resources: list[list[float]]
    ""

    """
    Caches the next available time and available resources for each resource type.

    Each resource maintains two lists in the following way:

    next_available_time[i] = [INF, t_1, ..., t_k]
    where t_1 > t_2 > ... > t_k are the times at which the available resources change.

    available_resources[i] = [C_i, r_1, ..., r_k]
    where r_j is the amount of available resources during the interval [t_{j-1}, t_j).

    So, when a task starts at time t, we insert its end time into the next_available_time list,
    and update the available_resources list accordingly.
    """
    next_available_time: list[list[Time]]
    available_resources: list[list[float]]

    def __init__(
        self, capacities: Iterable[Float], resource_usage: Iterable[str]
    ) -> None:
        self.capacities = convert_to_list(capacities, float)
        self.resource_tags = list(resource_usage)

        self.resources = []

        self.next_available_time = [[] for _ in self.resource_tags]
        self.available_resources = [[] for _ in self.resource_tags]

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.capacities, self.resource_tags),
            (
                self.resources,
                self.next_available_time,
                self.available_resources,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.resources, self.next_available_time, self.available_resources) = (
            state
        )

    def initialize(self, state: ScheduleState) -> None:
        self.resources = [
            convert_to_list(state.instance.task_instance[resource], float)
            for resource in self.resource_tags
        ]

    def reset(self, state: ScheduleState) -> None:
        for i, capacity in enumerate(self.capacities):
            self.next_available_time[i] = [MAX_TIME]
            self.available_resources[i] = [capacity]

        for capacity, task_demand in zip(self.capacities, self.resources):
            for task_id, demand in enumerate(task_demand):
                if demand > capacity:
                    state.forbid_task(task_id)

    def propagate(self, event: DomainEvent, state: ScheduleState) -> None:
        task_id = event.task_id

        if not event.is_assignment():
            return

        for i, task_resources in enumerate(self.resources):
            resource_usage = task_resources[task_id]

            if resource_usage <= 0:
                continue

            next_available_time = self.next_available_time[i]
            available_resources = self.available_resources[i]

            while next_available_time and next_available_time[-1] <= state.time:
                next_available_time.pop()
                available_resources.pop()

            end_time = state.get_end_lb(task_id)

            idx = binary_search(next_available_time, end_time, decreasing=True)

            self.next_available_time[i].insert(idx, end_time)
            self.available_resources[i].insert(
                idx, available_resources[idx - 1]
            )

            for j in range(idx, len(available_resources)):
                available_resources[j] -= resource_usage

        for i, task_resources in enumerate(self.resources):
            next_available_time = self.next_available_time[i]
            available_resources = self.available_resources[i]

            for other_task in list(state.runtime_state.awaiting_tasks):
                resource_usage = task_resources[other_task]

                if resource_usage <= 0:
                    continue

                idx = binary_search(
                    available_resources, resource_usage, decreasing=True
                )

                earliest_start = (
                    next_available_time[idx]
                    if idx < len(available_resources)
                    else state.time
                )

                state.tight_start_lb(other_task, earliest_start)


class NonRenewableResourceConstraint(Constraint):
    """
    Resource constraint for the scheduling environment.
    This constraint defines the non-renewable resources available for tasks and their usage.
    The resources can be defined as a list of capacities and a list of resource usage for each task.

    Arguments:
        capacities: Iterable[float]
            A list of capacities for each non-renewable resource. The length of the list should
            be equal to the number of resources.

        resource_usage: Iterable[str]
            A list of strings that define the resource usage for each task.
            Each string refers to a column in the tasks data that contains the
            resource usage for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    resource_tags: list[str]
    capacities: list[float]
    resources: list[list[float]]

    current_capacities: list[float]

    def __init__(
        self, capacities: Iterable[Float], resource_usage: Iterable[str]
    ) -> None:
        self.capacities = convert_to_list(capacities, float)
        self.resource_tags = list(resource_usage)

        self.current_capacities = self.capacities.copy()

        self.resources = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.capacities, self.resource_tags),
            (self.resources, self.current_capacities),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.resources, self.current_capacities) = state

    def initialize(self, state: ScheduleState) -> None:
        self.resources.clear()

        for resource in self.resource_tags:
            self.resources.append(
                convert_to_list(state.instance.task_instance[resource], float)
            )

    def reset(self, state: ScheduleState) -> None:
        self.current_capacities = self.capacities.copy()

    def propagate(self, event: DomainEvent, state: ScheduleState) -> None:
        task_id = event.task_id

        if not event.is_assignment():
            return

        for i, task_resources in enumerate(self.resources):
            resource_usage = task_resources[task_id]

            if resource_usage <= 0:
                continue

            self.current_capacities[i] -= resource_usage

            for other_task in list(state.runtime_state.awaiting_tasks):
                other_usage = task_resources[other_task]

                if other_usage <= 0:
                    continue

                if self.current_capacities[i] < other_usage:
                    state.forbid_task(other_task)

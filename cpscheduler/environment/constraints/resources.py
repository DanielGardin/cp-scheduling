from typing import Any, TypeVar, TYPE_CHECKING
from collections.abc import Iterable

from cpscheduler.environment.utils import convert_to_list

from cpscheduler.environment.constants import Time, Float, MAX_TIME
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint

if TYPE_CHECKING:
    from _typeshed import SupportsDunderLT

_SupportsLT = TypeVar("_SupportsLT", bound="SupportsDunderLT[Any]")


def binary_search(
    array: list[_SupportsLT],
    target: _SupportsLT,
    left: int = 0,
    right: int = -1,
    decreasing: bool = False,
) -> int:
    """
    Perform a binary search on a sorted array.

    Parameters
    ----------
    array: list
        The sorted array to be searched.

    target: Any
        The target value to be searched for.

    left: int, optional
        The left index of the search interval.

    right: int, optional
        The right index of the search interval.

    Returns
    -------
    int
        The index of the inclusion (to the right) of the target value in the array.
    """
    if right < 0:
        right = len(array) + right

    while left <= right:
        mid = (left + right) // 2

        if array[mid] == target:
            return mid + 1

        if array[mid] < target:
            if decreasing:
                right = mid - 1

            else:
                left = mid + 1

        else:
            if decreasing:
                left = mid + 1

            else:
                right = mid - 1

    return left


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

    def on_assignment(
        self, task_id: Time, machine_id: Time, state: ScheduleState
    ) -> None:
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

    def on_assignment(
        self, task_id: Time, machine_id: Time, state: ScheduleState
    ) -> None:
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

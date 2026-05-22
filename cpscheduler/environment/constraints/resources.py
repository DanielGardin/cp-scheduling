from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from cpscheduler.environment.constants import MAX_TIME, Float, Time
from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.instance import (
    UNSET,
    ProblemInstance,
    TaskFeature,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list

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
    __args__ = ("capacity",)

    resources: TaskFeature[float]
    capacity: float

    _next_available_time: list[Time]
    """Cache for the next available time:

    next_available_time = [MAX_TIME, t_1, ..., t_k],
    where t_1 > t_2 > ... > t_k are the times at which a task finishes its processing,
    releasing resource to other awaiting tasks.
    """

    _available_resources: list[float]
    """Cache for the remaining resources:

    available_resources = [capacity, r_1, ..., r_k]
    where r_j is the amount of available resources during the interval [t_j, t_{j-1}).
    """

    def __init__(
        self,
        capacity: float,
        resource_tag: str = "resource",
        resources: Iterable[Float] | None = None,
    ) -> None:
        self.capacity = capacity

        self.resources = TaskFeature(
            resource_tag,
            float,
            "cost",
            default=(
                convert_to_list(resources, float) if resources is not None else UNSET
            ),
        )

    def get_features(self) -> list[TaskFeature]:
        return [self.resources]

    def initialize(self, instance: ProblemInstance) -> None:
        self._next_available_time = []
        self._available_resources = []

    def reset(self, state: ScheduleState) -> None:
        self._next_available_time.clear()
        self._available_resources.clear()

        self._next_available_time.append(MAX_TIME)
        self._available_resources.append(self.capacity)

        capacity = self.capacity
        for task_id, resource in enumerate(self.resources.value):
            if resource > capacity:
                state.forbid_task(task_id)

    def on_assignment(
        self, task_id: Time, machine_id: Time, state: ScheduleState
    ) -> None:
        resources = self.resources.value
        resource_usage = resources[task_id]

        if resource_usage <= 0:
            return

        next_available_time = self._next_available_time
        available_resources = self._available_resources

        end_time = state.get_end_lb(task_id)

        idx = binary_search(next_available_time, end_time, decreasing=True)

        next_available_time.insert(idx, end_time)
        available_resources.insert(idx, available_resources[idx - 1])

        for j in range(idx, len(available_resources)):
            available_resources[j] -= resource_usage

        for other_task in state.runtime.get_awaiting_tasks():
            resource_usage = resources[other_task]

            if resource_usage <= 0:
                continue

            idx = binary_search(available_resources, resource_usage, decreasing=True)

            earliest_start = (
                next_available_time[idx]
                if idx < len(available_resources)
                else state.time
            )

            state.tight_start_lb(other_task, earliest_start)

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        next_available_time = self._next_available_time
        available_resources = self._available_resources

        current_time = state.time

        while next_available_time and next_available_time[-1] <= current_time:
            next_available_time.pop()
            available_resources.pop()


# TODO: Convert external information as Features
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

    resources: TaskFeature[float]
    capacity: float

    _current_capacity: float

    def __init__(
        self,
        capacity: float,
        resource_tag: str = "resource",
        resources: Iterable[Float] | None = None,
    ) -> None:
        self.capacity = capacity

        self.resources = TaskFeature(
            resource_tag,
            float,
            "cost",
            default=(
                convert_to_list(resources, float) if resources is not None else UNSET
            ),
        )

    def get_features(self) -> list[TaskFeature]:
        return [self.resources]

    def initialize(self, instance: ProblemInstance) -> None:
        self._current_capacity = self.capacity

    def reset(self, state: ScheduleState) -> None:
        self._current_capacity = self.capacity

    def on_assignment(
        self, task_id: Time, machine_id: Time, state: ScheduleState
    ) -> None:
        resources = self.resources.value
        resource_usage = resources[task_id]

        if resource_usage <= 0:
            return

        current_capacity = self._current_capacity - resource_usage
        self._current_capacity = current_capacity

        for other_task in state.runtime.get_awaiting_tasks():
            other_usage = resources[other_task]

            if other_usage <= 0:
                continue

            if current_capacity < other_usage:
                state.forbid_task(other_task)

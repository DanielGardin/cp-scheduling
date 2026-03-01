from typing import Any

from cpscheduler.utils.list_utils import convert_to_list

from cpscheduler.environment.constants import TASK_ID, TIME, Int
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint

class ReleaseDateConstraint(Constraint):
    """
    Release date constraint for the scheduling environment.

    This constraint defines the release dates for tasks, which are the earliest times
    that the tasks can start. The release dates can be defined as a mapping of task IDs
    to their respective release dates, or as a string that refers to a column in the tasks data.

    Arguments:
        release_dates: Mapping[int, int] | str
            A mapping of task IDs to their respective release dates. If a string is provided,
            it refers to a column in the tasks data that contains the release dates for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    release_tag: str
    release_dates: list[TIME]

    def __init__(self, release_dates: str = "release_time"):
        self.release_tag = release_dates

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.release_tag,),
            (self.release_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.release_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.release_dates = convert_to_list(
            state.instance.task_instance[self.release_tag], TIME
        )

    def reset(self, state: ScheduleState) -> None:
        for task_id, release_time in enumerate(self.release_dates):
            state.tight_start_lb(task_id, release_time)

    def get_entry(self) -> str:
        if self.release_dates:
            release_time = self.release_dates[TASK_ID(0)]

            for rt in self.release_dates:
                if rt != release_time:
                    return "r_j"

            return f"r_j={release_time}"

        return "r_j"


class DeadlineConstraint(Constraint):
    """
    Deadline constraint for the scheduling environment.

    This constraint defines the deadlines for tasks, which are the latest times
    that the tasks can be completed. The deadlines can be defined as a mapping of task IDs
    to their respective deadlines, or as a string that refers to a column in the tasks data.

    Arguments:
        deadlines: Mapping[int, int] | str
            A mapping of task IDs to their respective deadlines. If a string is provided,
            it refers to a column in the tasks data that contains the deadlines for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    due_tag: str
    due_dates: list[TIME]

    const_due: TIME | None

    def __init__(
        self, due_dates: str = "due_dates", const_due: Int | None = None
    ):
        self.due_tag = due_dates
        self.const_due = TIME(const_due) if const_due is not None else None

        self.due_dates = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.const_due),
            (self.due_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.due_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        if self.const_due is None:
            self.due_dates = convert_to_list(
                state.instance.task_instance[self.due_tag], TIME
            )

    def reset(self, state: ScheduleState) -> None:
        if self.const_due is not None:
            for task_id in range(state.n_tasks):
                state.tight_end_ub(task_id, self.const_due)

        else:
            for task_id, due_time in enumerate(self.due_dates):
                state.tight_end_ub(task_id, due_time)

    def get_entry(self) -> str:
        if self.const_due is not None:
            return f"d_j={self.const_due}"

        if self.due_dates:
            due_time = self.due_dates[0]

            for dt in self.due_dates[1:]:
                if dt != due_time:
                    return "d_j"

            return f"d_j={due_time}"

        return "d_j"

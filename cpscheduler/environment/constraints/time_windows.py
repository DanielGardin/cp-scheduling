from cpscheduler.environment.utils.general import convert_to_list

from cpscheduler.environment.constants import Time, Int, MAX_TIME
from cpscheduler.environment.instance import (
    GlobalFeature, TaskFeature, UNSET
)
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint


class HorizonConstraint(Constraint):
    """
    Horizon constraint for the scheduling environment.

    Defines a hard upper bound on the completion time of all tasks.

    This is technically equivalent to a deadline constraint with a constant
    deadline for all tasks, but differs on the semantic level, as it does not
    produce a standalone entry in the schedule.

    Arguments:
        horizon: int
            The upper bound on the completion time of all tasks.
    """

    horizon: GlobalFeature[Time]

    def __init__(self, horizon: Int = MAX_TIME):
        self.horizon = GlobalFeature(
            name="horizon",
            pytype=Time,
            semantic="time",
            default=Time(horizon)
        )

    def set_horizon(self, horizon: Int) -> None:
        self.horizon.set_data(Time(horizon))

    def get_features(self) -> list[GlobalFeature]:
        return [self.horizon]

    def reset(self, state: ScheduleState) -> None:
        horizon = self.horizon.value

        for task_id in range(state.n_tasks):
            state.tight_end_ub(task_id, horizon)


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
    """

    release_dates: TaskFeature[Time]

    def __init__(
        self,
        release_tag: str = "release_time",
        release_dates: list[Int] | None = None
    ):
        self.release_dates = TaskFeature(
            name=release_tag,
            elem_type=Time,
            semantic="time",
            default=(
                convert_to_list(release_dates, Time)
                if release_dates is not None else UNSET
            )
        )

    def get_features(self) -> list[TaskFeature]:
        return [self.release_dates]

    def reset(self, state: ScheduleState) -> None:
        for task_id, release_time in enumerate(self.release_dates.value):
            state.tight_start_lb(task_id, release_time)

    @classmethod
    def get_general_entry(cls) -> str:
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

    due_dates: TaskFeature[Time]

    def __init__(
        self,
        due_tag: str = "due_time",
        due_dates: list[Int] | None = None
    ):
        self.due_dates = TaskFeature(
            name=due_tag,
            elem_type=Time,
            semantic="time",
            default=(
                convert_to_list(due_dates, Time)
                if due_dates is not None else UNSET
            )
        )

    def get_features(self) -> list[TaskFeature]:
        return [self.due_dates]

    def reset(self, state: ScheduleState) -> None:
        for task_id, due_time in enumerate(self.due_dates.value):
            state.tight_end_ub(task_id, due_time)

    @classmethod
    def get_general_entry(cls) -> str:
        return "d_j"

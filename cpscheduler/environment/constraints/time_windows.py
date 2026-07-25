"""Time window constraints for the scheduling environment."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import MAX_TIME, Int, Time
from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.instance import Feature
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class HorizonConstraint(Constraint):
    """Horizon constraint for the scheduling environment.

    Defines a hard upper bound on the completion time of all tasks.

    This is technically equivalent to a deadline constraint with a constant
    deadline for all tasks, but differs on the semantic level, as it does not
    produce a standalone entry in the schedule.
    """

    horizon: Feature[Time]

    def __init__(self, horizon: Int = MAX_TIME):
        """Initialize the Horizon Constraint.

        Parameters
        ----------
        horizon: int
            The upper bound on the completion time of all tasks.

        """
        self.horizon = Feature(name="horizon", shape=(), value_type="time")
        self.horizon.own_data(Time(horizon))

    def set_horizon(self, horizon: Int) -> None:
        """Set the horizon value."""
        self.horizon.own_data(Time(horizon))

    @override
    def reset(self, state: ScheduleState) -> None:
        horizon = self.horizon.value

        for task_id in range(state.n_tasks):
            state.tight_end_ub(task_id, horizon)


class ReleaseDateConstraint(Constraint):
    """Release date constraint for the scheduling environment.

    This constraint defines the release dates for tasks, which are the earliest times
    that the tasks can start. The release dates can be defined as a mapping of task IDs
    to their respective release dates, or as a string that refers to a column in the tasks data.

    Arguments:
        release_dates: Mapping[int, int] | str
            A mapping of task IDs to their respective release dates. If a string is provided,
            it refers to a column in the tasks data that contains the release dates for each task.

    """

    release_dates: Feature[list[Time]]

    def __init__(
        self,
        release_tag: str = "release_time",
        release_dates: Iterable[Int] | None = None,
    ):
        """Initialize the Release Date Constraint.

        Parameters
        ----------
        release_tag: str, optional
            The tag for the release dates feature.

        release_dates: list[int] | None, optional
            A list of release dates for each task.
            If None, release dates must be provided in the instance data.
            Default to None.

        """
        self.release_dates = Feature(
            name=release_tag,
            preprocess=self._load_dates,
            shape=("n_tasks",),
            value_type="time",
        )

        if release_dates is not None:
            self.release_dates.own_data(release_dates)

    def _load_dates(self, dates: Iterable[Int]) -> list[Time]:
        return convert_to_list(dates, Time)

    @override
    def reset(self, state: ScheduleState) -> None:
        for task_id, release_time in enumerate(self.release_dates.value):
            state.tight_start_lb(task_id, release_time)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "r_j"


class DeadlineConstraint(Constraint):
    """Deadline constraint for the scheduling environment.

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

    due_dates: Feature[list[Time]]

    def __init__(
        self, due_tag: str = "due_time", due_dates: list[Int] | None = None
    ):
        """Initialize the Deadline Constraint.

        Parameters
        ----------
        due_tag: str, optional
            The tag for the deadlines feature.

        due_dates: list[int] | None, optional
            A list of deadlines for each task.
            If None, deadlines must be provided in the instance data.

        """
        self.due_dates = Feature(
            name=due_tag,
            preprocess=self._load_dates,
            shape=("n_tasks",),
            value_type="time",
        )

    def _load_dates(self, dates: Iterable[Int]) -> list[Time]:
        return convert_to_list(dates, Time)

    @override
    def reset(self, state: ScheduleState) -> None:
        for task_id, due_time in enumerate(self.due_dates.value):
            state.tight_end_ub(task_id, due_time)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "d_j"

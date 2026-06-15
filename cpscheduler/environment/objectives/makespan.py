"""Makespan and maximum lateness objectives."""

from typing_extensions import override

from cpscheduler.environment.constants import MAX_TIME, MachineID, TaskID, Time
from cpscheduler.environment.instance import JobFeature
from cpscheduler.environment.objectives.base import Objective
from cpscheduler.environment.state import ScheduleState


class Makespan(Objective):
    """Makespan objective.

    This objective function aims to minimize the time at which all tasks are completed.
    """

    _value: Time

    @property
    @override
    def regular(self) -> bool:
        return True

    @override
    def reset(self, state: ScheduleState) -> None:
        super().reset(state)

        self._value = 0

    @override
    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._value = max(self._value, state.get_end(task_id))

    @override
    def get_current(self, state: ScheduleState) -> float:
        return float(self._value)

    @override
    def __call__(self, state: ScheduleState) -> float:
        completed_tasks = state.get_completed_tasks()

        if not completed_tasks:
            return 0.0

        return float(max(state.get_end(task_id) for task_id in completed_tasks))

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "C_max"


class MaximumLateness(Objective):
    """Maximum Lateness objective.

    This objective function aims to minimize the maximum lateness of all jobs.
    Lateness of a job is defined as the amount of time by which its completion time
    exceeds its due date, i.e., L_j = C_j - d_j
    """

    _value: Time

    due_dates: JobFeature[Time]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        """Initialize the Maximum Lateness objective.

        Parameters
        ----------
        due_dates: str, optional
            The name of the job feature that contains the due dates.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.due_dates = JobFeature(name=due_dates, semantic="time", shape=())

    @property
    @override
    def regular(self) -> bool:
        return True

    @override
    def get_features(self) -> list[JobFeature]:
        return [self.due_dates]

    @override
    def reset(self, state: ScheduleState) -> None:
        super().reset(state)
        self._value = -MAX_TIME

    @override
    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        d_j = self.due_dates.value[job_id]

        self._value = max(self._value, state.get_end(task_id) - d_j)

    @override
    def get_current(self, state: ScheduleState) -> float:
        return float(self._value)

    @override
    def __call__(self, state: ScheduleState) -> float:
        completed_tasks = state.get_completed_tasks()

        if not completed_tasks:
            return float(-MAX_TIME)

        job_ids = state.instance.job_ids
        due_dates = self.due_dates.value

        return float(
            max(
                state.get_end(task_id) - due_dates[job_ids[task_id]]
                for task_id in completed_tasks
            )
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "L_max"

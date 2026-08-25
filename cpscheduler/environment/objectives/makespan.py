"""Makespan and maximum lateness objectives."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import (
    MAX_TIME,
    Int,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.instance import Feature
from cpscheduler.environment.objectives.base import Objective
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class Makespan(Objective):
    """Makespan objective.

    This objective function aims to minimize the time at which all tasks are completed.
    """

    _value: Time
    _lb: Time
    _ub: Time

    @property
    @override
    def regular(self) -> bool:
        return True

    @override
    def reset(self, state: ScheduleState) -> None:
        super().reset(state)

        self._value = 0
        self._lb = 0
        self._ub = 0

    @property
    @override
    def lb(self) -> float:
        return float(self._lb)

    @property
    @override
    def ub(self) -> float:
        return float(self._ub)

    @property
    @override
    def value(self) -> float:
        return float(self._value)

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._value = max(self._value, state.get_end(task_id))

    @override
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._lb = max(self._lb, state.get_end_lb(task_id))

    @override
    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._ub = max(self._ub, state.get_end_ub(task_id))

    @override
    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._lb = max(self._lb, state.get_end_lb(task_id))

    @override
    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._ub = max(self._ub, state.get_end_ub(task_id))

    @override
    def compute(self, state: ScheduleState) -> float:
        completed_tasks = state.get_assigned_tasks()

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
    _lb: Time
    _ub: Time

    due_dates: Feature[list[Time]]

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
        minimize: bool = True,
    ):
        """Initialize the Maximum Lateness objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: Iterable[Time] | None, optional
            The due dates for each job.
            If None, due dates must be provided in the instance data.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.due_dates = Feature(
            name=due_dates_tag,
            preprocess=self._load_dates,
            shape=("n_jobs",),
            value_type="time",
        )

        if due_dates is not None:
            self.due_dates.own_data(due_dates)

    def _load_dates(self, dates: Iterable[Int]) -> list[Time]:
        return convert_to_list(dates, Time)

    @property
    @override
    def regular(self) -> bool:
        return True

    @property
    @override
    def lb(self) -> float:
        return float(self._lb)

    @property
    @override
    def ub(self) -> float:
        return float(self._ub)

    @property
    @override
    def value(self) -> float:
        return float(self._value)

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        d_j = self.due_dates.value[job_id]

        self._value = max(self._value, state.get_end(task_id) - d_j)

    @override
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        d_j = self.due_dates.value[job_id]

        self._lb = max(self._lb, state.get_end_lb(task_id) - d_j)

    @override
    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        d_j = self.due_dates.value[job_id]

        self._ub = max(self._ub, state.get_end_ub(task_id) - d_j)

    @override
    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        d_j = self.due_dates.value[job_id]

        self._lb = max(self._lb, state.get_end_lb(task_id) - d_j)

    @override
    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        d_j = self.due_dates.value[job_id]

        self._ub = max(self._ub, state.get_end_ub(task_id) - d_j)

    @override
    def compute(self, state: ScheduleState) -> float:
        completed_tasks = state.get_assigned_tasks()

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

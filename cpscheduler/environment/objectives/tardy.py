from typing import Any

from cpscheduler.environment.utils import convert_to_list
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import Objective
from cpscheduler.environment.objectives.makespan import makespan_


class TotalTardyJobs(Objective):
    """
    The total tardy jobs objective function, which aims to minimize the number
    of tardy jobs.
    A job is tardy if its completion time exceeds its due date.
    """

    _tardy_jobs: set[TaskID]

    due_tag: str
    due_dates: list[Time]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self._tardy_jobs = set()

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.minimize),
            (self._tardy_jobs, self.due_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._tardy_jobs, self.due_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], Time
        )

    def reset(self, state: ScheduleState) -> None:
        self._tardy_jobs.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        due_date = self.due_dates[job_id]

        if state.time > due_date:
            self._tardy_jobs.add(job_id)

    def get_current(self, state: ScheduleState) -> float:
        return float(len(self._tardy_jobs))

    def __call__(self, state: ScheduleState) -> float:
        due_dates = self.due_dates

        return float(
            sum(
                1
                for due_date, tasks in zip(due_dates, state.instance.job_tasks)
                if makespan_(state, tasks) > float(due_date)
            )
        )

    def get_entry(self) -> str:
        return "ΣU_j"


class WeightedTardyJobs(Objective):
    """
    The weighted tardy jobs objective function, which aims to minimize the
    weighted number of tardy jobs.
    A job is tardy if its completion time exceeds its due date.
    """

    _weighted_job_tardy: dict[TaskID, float]

    weights_tag: str
    job_weights: list[float]

    due_tag: str
    due_dates: list[Time]

    def __init__(
        self,
        due_dates: str = "due_date",
        job_weights: str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)

        self.due_tag = due_dates
        self.weight_tag = job_weights

        self._weighted_job_tardy = {}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.weight_tag, self.minimize),
            (self._weighted_job_tardy, self.due_dates, self.job_weights),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._weighted_job_tardy, self.due_dates, self.job_weights) = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], Time
        )

        self.job_weights = convert_to_list(
            state.instance.task_instance[self.weight_tag], float
        )

    def reset(self, state: ScheduleState) -> None:
        self._weighted_job_tardy.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        due_date = self.due_dates[job_id]
        weight = self.job_weights[job_id]

        if state.time > due_date:
            self._weighted_job_tardy[job_id] = weight

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._weighted_job_tardy.values()))

    def __call__(self, state: ScheduleState) -> float:
        due_dates = self.due_dates
        job_weights = self.job_weights

        return float(
            sum(
                job_weights[j]
                for j, (due_date, tasks) in enumerate(
                    zip(due_dates, state.instance.job_tasks)
                )
                if makespan_(state, tasks) > float(due_date)
            )
        )

    def get_entry(self) -> str:
        return "Σw_jU_j"
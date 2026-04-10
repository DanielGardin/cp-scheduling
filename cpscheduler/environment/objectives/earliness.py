from typing import Any

from cpscheduler.environment.utils import convert_to_list
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import Objective
from cpscheduler.environment.objectives.makespan import makespan_


class TotalEarliness(Objective):
    """
    The total earliness objective function, which aims to minimize the sum of
    earliness of all tasks.
    Earliness is defined as the difference between the due date and the
    completion time, if the task is completed early.
    """

    _job_earliness: dict[TaskID, Time]

    due_tag: str
    due_dates: list[Time]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self._job_earliness = {}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.minimize),
            (self._job_earliness, self.due_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._job_earliness, self.due_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], Time
        )

    def reset(self, state: ScheduleState) -> None:
        self._job_earliness.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        due_date = self.due_dates[job_id]

        earliness = due_date - state.time
        if earliness > 0:
            self._job_earliness[job_id] = earliness

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._job_earliness.values()))

    def __call__(self, state: ScheduleState) -> float:
        due_dates = self.due_dates

        return sum(
            max(float(due_date) - makespan_(state, tasks), 0.0)
            for due_date, tasks in zip(due_dates, state.instance.job_tasks)
        )

    def get_entry(self) -> str:
        return "ΣE_j"


class WeightedEarliness(Objective):
    """
    The weighted earliness objective function, which aims to minimize the weighted sum of earliness
    of all tasks. Earliness is defined as the difference between the due date and the completion time,
    if the task is completed early.
    """

    _weighted_job_earliness: dict[TaskID, float]

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

        self._weighted_job_earliness = {}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.weight_tag, self.minimize),
            (self._weighted_job_earliness, self.due_dates, self.job_weights),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._weighted_job_earliness, self.due_dates, self.job_weights) = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], Time
        )

        self.job_weights = convert_to_list(
            state.instance.task_instance[self.weight_tag], float
        )

    def reset(self, state: ScheduleState) -> None:
        self._weighted_job_earliness.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        due_date = self.due_dates[job_id]
        weight = self.job_weights[job_id]

        earliness = due_date - state.time
        if earliness > 0:
            self._weighted_job_earliness[job_id] = weight * float(earliness)

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._weighted_job_earliness.values()))

    def __call__(self, state: ScheduleState) -> float:
        due_dates = self.due_dates
        job_weights = self.job_weights

        return sum(
            job_weights[j] * max(float(due_date) - makespan_(state, tasks), 0.0)
            for j, (due_date, tasks) in enumerate(
                zip(due_dates, state.instance.job_tasks)
            )
        )

    def get_entry(self) -> str:
        return "Σw_jE_j"
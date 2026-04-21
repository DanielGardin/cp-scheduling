from cpscheduler.environment.utils import convert_to_list
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import Objective, RegularObjective
from cpscheduler.environment.objectives.makespan import makespan_


class MaximumLateness(RegularObjective):
    """
    The maximum lateness objective function, which aims to minimize the maximum
    lateness of all tasks.
    Lateness is defined as the difference between the completion time and the
    due date.
    """

    _job_lateness: dict[TaskID, Time]

    due_tag: str
    due_dates: list[Time]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self._job_lateness = {}

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], Time
        )

    def reset(self, state: ScheduleState) -> None:
        self._job_lateness.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        due_date = self.due_dates[job_id]

        lateness = state.time - due_date
        if lateness > 0:
            self._job_lateness[job_id] = lateness

    def get_current(self, state: ScheduleState) -> float:
        return float(max(self._job_lateness.values(), default=0.0))

    def __call__(self, state: ScheduleState) -> float:
        due_dates = self.due_dates

        maximum_lateness = 0.0

        for due_date, tasks in zip(due_dates, state.instance.job_tasks):
            completion_time = makespan_(state, tasks)
            lateness = completion_time - float(due_date)

            if lateness > maximum_lateness:
                maximum_lateness = lateness

        return maximum_lateness

    def get_entry(self) -> str:
        return "L_max"


class TotalTardiness(RegularObjective):
    """
    The total tardiness objective function, which aims to minimize the sum of
    tardiness of all tasks.
    Tardiness is defined as the difference between the completion time and the
    due date, if the task is completed late.
    """

    _job_lateness: dict[TaskID, Time]

    due_tag: str
    due_dates: list[Time]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self._job_lateness = {}


    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], Time
        )

    def reset(self, state: ScheduleState) -> None:
        self._job_lateness.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        due_date = self.due_dates[job_id]

        lateness = state.time - due_date
        if lateness > 0:
            self._job_lateness[job_id] = lateness

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._job_lateness.values()))

    def __call__(self, state: ScheduleState) -> float:
        due_dates = self.due_dates

        return sum(
            max(makespan_(state, tasks) - float(due_date), 0.0)
            for due_date, tasks in zip(due_dates, state.instance.job_tasks)
        )

    def get_entry(self) -> str:
        return "ΣT_j"


class WeightedTardiness(Objective):
    """
    The weighted tardiness objective function, which aims to minimize the weighted sum of tardiness
    of all tasks. Tardiness is defined as the difference between the completion time and the due
    date, if the task is completed late.
    """

    _weighted_job_lateness: dict[TaskID, float]

    weight_tag: str
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
        self._weighted_job_lateness = {}

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], Time
        )

        self.job_weights = convert_to_list(
            state.instance.task_instance[self.weight_tag], float
        )

    def reset(self, state: ScheduleState) -> None:
        self._weighted_job_lateness.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        due_date = self.due_dates[job_id]
        weight = self.job_weights[job_id]

        lateness = state.time - due_date
        if lateness > 0:
            self._weighted_job_lateness[job_id] = weight * float(lateness)

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._weighted_job_lateness.values()))

    def __call__(self, state: ScheduleState) -> float:
        due_dates = self.due_dates

        return sum(
            max(makespan_(state, tasks) - float(due_date), 0.0)
            for due_date, tasks in zip(due_dates, state.instance.job_tasks)
        )

    def get_entry(self) -> str:
        return "Σw_jT_j"

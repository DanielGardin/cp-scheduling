from cpscheduler.environment.constants import MAX_TIME, MachineID, TaskID, Time

from cpscheduler.environment.instance import JobFeature
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import Objective

class Makespan(Objective):
    """
    Classic makespan objective function, which aims to minimize the time at
    which all tasks are completed.
    """

    _value: Time

    @property
    def regular(self) -> bool:
        return True

    def reset(self, state: ScheduleState) -> None:
        super().reset(state)

        self._value = 0

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._value = max(self._value, state.get_end(task_id))

    def get_current(self, state: ScheduleState) -> float:
        return float(self._value)

    def __call__(self, state: ScheduleState) -> float:
        completed_tasks = state.get_completed_tasks()

        if not completed_tasks:
            return 0.0

        return float(max(
            state.get_end(task_id)
            for task_id in completed_tasks
        ))

    @classmethod
    def get_general_entry(cls) -> str:
        return "C_max"

class MaximumLateness(Objective):
    """
    Classic makespan objective function, which aims to minimize the time at
    which all tasks are completed.
    """

    _value: Time

    due_dates: JobFeature[Time]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)

        self.due_dates = JobFeature(
            name=due_dates,
            elem_type=Time,
            semantic="time"
        )

    @property
    def regular(self) -> bool:
        return True

    def get_features(self) -> list[JobFeature]:
        return [self.due_dates]

    def reset(self, state: ScheduleState) -> None:
        super().reset(state)
        self._value = -MAX_TIME

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        d_j = self.due_dates.value[job_id]

        self._value = max(self._value, state.get_end(task_id) - d_j)

    def get_current(self, state: ScheduleState) -> float:
        return float(self._value)

    def __call__(self, state: ScheduleState) -> float:
        completed_tasks = state.get_completed_tasks()

        if not completed_tasks:
            return float(-MAX_TIME)

        job_ids = state.instance.job_ids
        due_dates = self.due_dates.value

        return float(max(
            state.get_end(task_id) - due_dates[job_ids[task_id]]
            for task_id in completed_tasks
        ))

    @classmethod
    def get_general_entry(cls) -> str:
        return "L_max"

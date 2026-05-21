from cpscheduler.environment.constants import Time

from cpscheduler.environment.instance import JobFeature
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import RegularObjective

class TotalTardiness(RegularObjective):
    """
    The total tardiness objective function, which aims to minimize the sum of
    tardiness of all tasks.
    Tardiness is defined as the difference between the completion time and the
    due date, if the task is completed late.
    """

    due_dates: JobFeature[Time]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True
    ) -> None:
        super().__init__(minimize)

        self.due_dates = JobFeature(
            name=due_dates,
            elem_type=Time,
            semantic="time"
        )

    def get_features(self) -> list[JobFeature]:
        return [self.due_dates]

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(
            max(C_j - d_j, 0)
            for d_j, C_j in zip(
                self.due_dates.value, self._job_completion
            )
        ))

    def __call__(self, state: ScheduleState) -> float:
        return float(sum(
            max(C_j - d_j, 0)
            for d_j, C_j in zip(
                self.due_dates.value, self.completion_times(state)
            )
        ))

    @classmethod
    def get_general_entry(cls) -> str:
        return "ΣT_j"


class WeightedTardiness(TotalTardiness):
    """
    The weighted tardiness objective function, which aims to minimize the weighted sum of tardiness
    of all tasks. Tardiness is defined as the difference between the completion time and the due
    date, if the task is completed late.
    """
    
    weights: JobFeature[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        job_weights: str = "weight",
        minimize: bool = True,
    ):
        super().__init__(due_dates, minimize)

        self.weights = JobFeature(
            name=job_weights,
            elem_type=float,
            semantic="continuous",
        )

    @property
    def regular(self) -> bool:
        return all(weight >= 0 for weight in self.weights.value)

    def get_features(self) -> list[JobFeature]:
        return [self.due_dates, self.weights]

    def get_current(self, state: ScheduleState) -> float:
        return sum(
            w_j * float(max(C_j - d_j, 0))
            for w_j, d_j, C_j in zip(
                self.weights.value, self.due_dates.value, self._job_completion
            )
        )

    def __call__(self, state: ScheduleState) -> float:
        return float(sum(
            w_j * float(max(C_j - d_j, 0))
            for w_j, d_j, C_j in zip(
                self.weights.value, self.due_dates.value, self.completion_times(state)
            )
        ))

    @classmethod
    def get_general_entry(cls) -> str:
        return "Σw_jT_j"

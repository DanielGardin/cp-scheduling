from cpscheduler.environment.constants import Time

from cpscheduler.environment.instance import JobFeature
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import CompletionTimeObjective


class TotalEarliness(CompletionTimeObjective):
    """
    The total earliness objective function, which aims to minimize the sum of
    earliness of all jobs.
    Earliness is defined as the difference between the due date and the
    completion time, if the job is completed early.
    """

    due_dates: JobFeature[Time]

    def __init__(
        self, due_dates: str = "due_date", minimize: bool = True
    ) -> None:
        super().__init__(minimize)

        self.due_dates = JobFeature(
            name=due_dates, elem_type=Time, semantic="time"
        )

    def get_features(self) -> list[JobFeature]:
        return [self.due_dates]

    def get_current(self, state: ScheduleState) -> float:
        return float(
            sum(
                max(d_j - C_j, 0)
                for d_j, C_j in zip(self.due_dates.value, self._job_completion)
            )
        )

    def __call__(self, state: ScheduleState) -> float:
        return float(
            sum(
                max(d_j - C_j, 0)
                for d_j, C_j in zip(
                    self.due_dates.value, self.completion_times(state)
                )
            )
        )

    @classmethod
    def get_general_entry(cls) -> str:
        return "ΣE_j"


class WeightedEarliness(TotalEarliness):
    """
    The weighted earliness objective function, which aims to minimize the
    weighted sum of earliness of all jobs.
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
        return float(
            sum(
                w_j * float(max(d_j - C_j, 0))
                for w_j, d_j, C_j in zip(
                    self.weights.value,
                    self.due_dates.value,
                    self._job_completion,
                )
            )
        )

    def __call__(self, state: ScheduleState) -> float:
        return float(
            sum(
                w_j * float(max(d_j - C_j, 0))
                for w_j, d_j, C_j in zip(
                    self.weights.value,
                    self.due_dates.value,
                    self.completion_times(state),
                )
            )
        )

    @classmethod
    def get_general_entry(cls) -> str:
        return "Σw_jE_j"

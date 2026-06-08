"""Earliness-based objectives."""

from typing import override

from cpscheduler.environment.constants import Time
from cpscheduler.environment.instance import JobFeature
from cpscheduler.environment.objectives.base import CompletionTimeObjective
from cpscheduler.environment.state import ScheduleState


class TotalEarliness(CompletionTimeObjective):
    """Total Earliness objective.

    This objective function aims to minimize the sum of earliness of all jobs.
    Earliness of a job is defined as the amount of time by which its completion time
    is earlier than its due date, i.e., E_j = max(d_j - C_j, 0)
    """

    due_dates: JobFeature[Time]

    def __init__(
        self, due_dates: str = "due_date", minimize: bool = True
    ) -> None:
        """Initialize the Total Earliness objective.

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

    @override
    def get_features(self) -> list[JobFeature]:
        return [self.due_dates]

    @override
    def get_current(self, state: ScheduleState) -> float:
        return float(
            sum(
                max(d_j - C_j, 0)
                for d_j, C_j in zip(
                    self.due_dates.value, self._job_completion, strict=False
                )
            )
        )

    @override
    def __call__(self, state: ScheduleState) -> float:
        return float(
            sum(
                max(d_j - C_j, 0)
                for d_j, C_j in zip(
                    self.due_dates.value,
                    self.completion_times(state),
                    strict=False,
                )
            )
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "ΣE_j"


class WeightedEarliness(TotalEarliness):
    """Weighted Earliness objective.

    This objective function aims to minimize the weighted sum of earliness of all jobs.
    Earliness of a job is defined as the amount of time by which its completion time
    is earlier than its due date, i.e., E_j = max(d_j - C_j, 0).
    The weighted variant optimizes Σw_jE_j, where w_j is the weight of job j.
    """

    weights: JobFeature[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        job_weights: str = "weight",
        minimize: bool = True,
    ):
        """Initialize the Weighted Earliness objective.

        Parameters
        ----------
        due_dates: str, optional
            The name of the job feature that contains the due dates.

        job_weights: str, optional
            The name of the job feature that contains the weights for each job.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(due_dates, minimize)

        self.weights = JobFeature(
            name=job_weights,
            semantic="continuous",
        )

    @property
    @override
    def regular(self) -> bool:
        return all(weight >= 0 for weight in self.weights.value)

    @override
    def get_features(self) -> list[JobFeature]:
        return [self.due_dates, self.weights]

    @override
    def get_current(self, state: ScheduleState) -> float:
        return float(
            sum(
                w_j * float(max(d_j - C_j, 0))
                for w_j, d_j, C_j in zip(
                    self.weights.value,
                    self.due_dates.value,
                    self._job_completion,
                    strict=False,
                )
            )
        )

    @override
    def __call__(self, state: ScheduleState) -> float:
        return float(
            sum(
                w_j * float(max(d_j - C_j, 0))
                for w_j, d_j, C_j in zip(
                    self.weights.value,
                    self.due_dates.value,
                    self.completion_times(state),
                    strict=False,
                )
            )
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Σw_jE_j"

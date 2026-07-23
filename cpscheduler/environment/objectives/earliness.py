"""Earliness-based objectives."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import Int, Time
from cpscheduler.environment.instance import Feature
from cpscheduler.environment.objectives.base import CompletionTimeObjective
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class TotalEarliness(CompletionTimeObjective):
    """Total Earliness objective.

    This objective function aims to minimize the sum of earliness of all jobs.
    Earliness of a job is defined as the amount of time by which its completion time
    is earlier than its due date, i.e., E_j = max(d_j - C_j, 0)
    """

    due_dates: Feature[list[Time]]

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
        minimize: bool = True,
    ) -> None:
        """Initialize the Total Earliness objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: Iterable[Time] | None, optional
            The due dates for each job.
            If None is provided, the due dates will be loaded from the instance.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.due_dates = Feature(
            name=due_dates_tag,
            preprocess=self._load_due_dates,
            shape=("n_jobs",),
        )

        if due_dates is not None:
            self.due_dates.own_data(due_dates)

    def _load_due_dates(self, due_dates: Iterable[Time]) -> list[Time]:
        return convert_to_list(due_dates, Time)

    @override
    def get_features(self) -> list[Feature]:
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

    weights: Feature[list[float]]

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
        weights_tag: str = "weight",
        weights: Iterable[float] | None = None,
        minimize: bool = True,
    ):
        """Initialize the Weighted Earliness objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: Iterable[Time] | None, optional
            The due dates for each job.
            If None is provided, the due dates will be loaded from the instance.

        weights_tag: str, optional
            The name of the job feature that contains the weights for each job.

        weights: str, optional
            The name of the job feature that contains the weights for each job.
            If None is provided, the weights will be loaded from the instance.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(due_dates_tag, due_dates, minimize)

        self.weights = Feature(
            name=weights_tag, preprocess=self._load_weights, shape=("n_jobs",)
        )

        if weights is not None:
            self.weights.own_data(weights)

    def _load_weights(self, weights: Iterable[float]) -> list[float]:
        return convert_to_list(weights, float)

    @property
    @override
    def regular(self) -> bool:
        return all(weight >= 0 for weight in self.weights.value)

    @override
    def get_features(self) -> list[Feature]:
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

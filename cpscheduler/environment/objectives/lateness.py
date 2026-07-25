"""Lateness-based objectives."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import Int, Time
from cpscheduler.environment.instance import Feature
from cpscheduler.environment.objectives.base import RegularObjective
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class TotalTardiness(RegularObjective):
    """Total Tardiness objective.

    This objective function aims to minimize the sum of tardiness of all jobs.
    Tardiness of a job is defined as the amount of time by which its completion time
    exceeds its due date, i.e., T_j = max(C_j - d_j, 0)
    """

    due_dates: Feature[list[Time]]

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
        minimize: bool = True,
    ) -> None:
        """Initialize the Total Tardiness objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: list[Time] | None, optional
            The due dates for each job.
            If None is provided, the due dates will be loaded from the instance.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.due_dates = Feature(
            name=due_dates_tag,
            shape=("n_jobs",),
            value_type="time",
            preprocess=self._load_due_dates,
        )

    def _load_due_dates(self, due_dates: Iterable[Int]) -> list[Time]:
        return convert_to_list(due_dates, Time)

    @override
    def get_current(self, state: ScheduleState) -> float:
        return float(
            sum(
                max(C_j - d_j, 0)
                for d_j, C_j in zip(
                    self.due_dates.value, self._job_completion, strict=False
                )
            )
        )

    @override
    def __call__(self, state: ScheduleState) -> float:
        return float(
            sum(
                max(C_j - d_j, 0)
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
        return "ΣT_j"


class WeightedTardiness(TotalTardiness):
    """Weighted Tardiness objective.

    This objective function aims to minimize the weighted sum of tardiness of all jobs.
    Tardiness of a job is defined as the amount of time by which its completion time
    exceeds its due date, i.e., T_j = max(C_j - d_j, 0).
    The weighted variant optimizes Σw_jT_j, where w_j is the weight of job j.
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
        """Initialize the Weighted Tardiness objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.
            Default is "due_date".

        due_dates: Iterable[Int] | None, optional
            The due dates for each job.
            If None is provided, the due dates will be loaded from the instance.

        weights_tag: str, optional
            The name of the job feature that contains the weights.
            Default is "weight".

        weights: Iterable[float] | None, optional
            The weights for each job.
            If None is provided, the weights will be loaded from the instance.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(due_dates_tag, due_dates, minimize)

        self.weights = Feature(
            name=weights_tag,
            preprocess=self._load_weights,
            shape=("n_jobs",),
            value_type="cost",
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
    def get_current(self, state: ScheduleState) -> float:
        return sum(
            w_j * float(max(C_j - d_j, 0))
            for w_j, d_j, C_j in zip(
                self.weights.value,
                self.due_dates.value,
                self._job_completion,
                strict=False,
            )
        )

    @override
    def __call__(self, state: ScheduleState) -> float:
        return float(
            sum(
                w_j * float(max(C_j - d_j, 0))
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
        return "Σw_jT_j"

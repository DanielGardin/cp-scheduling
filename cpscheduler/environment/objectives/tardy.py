"""Tardy jobs objectives."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import (
    Float,
    Int,
    Time,
)
from cpscheduler.environment.instance import Feature, ProblemInstance
from cpscheduler.environment.objectives.base import RegularObjective
from cpscheduler.environment.utils.general import convert_to_list


class TotalTardyJobs(RegularObjective):
    """Total Tardy Jobs objective.

    This objective function aims to minimize the number of tardy jobs.
    A job is tardy if its completion time exceeds its due date, i.e., C_j > d_j.
    """

    due_dates: Feature[list[Time]]

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
        minimize: bool = True,
    ) -> None:
        """Initialize the Total Tardy Jobs objective.

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
            preprocess=self._load_due_dates,
            shape=("n_jobs",),
            value_type="time",
        )

    def _load_due_dates(self, due_dates: Iterable[Int]) -> list[Time]:
        return convert_to_list(due_dates, Time)

    @override
    def evaluate(self, job_completions: list[Time]) -> float:
        return float(
            sum(
                C_j > d_j
                for d_j, C_j in zip(
                    self.due_dates.value, job_completions, strict=True
                )
            )
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "ΣU_j"


class WeightedTardyJobs(TotalTardyJobs):
    """Weighted Tardy Jobs objective.

    This objective function aims to minimize the weighted number of tardy jobs.
    A job is tardy if its completion time exceeds its due date, i.e., C_j > d_j.
    The weighted variant optimizes Σw_jU_j, where w_j is the weight of job j.
    """

    weights: Feature[list[float]]

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
        weights_tag: str = "weight",
        weights: Iterable[Float] | None = None,
        minimize: bool = True,
    ) -> None:
        """Initialize the Weighted Tardy Jobs objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: Iterable[Time] | None, optional
            The due dates for each job.
            If None, due dates must be provided in the instance data.

        weights_tag: str, optional
            The name of the job feature that contains the weights.
            Default to "weight".

        weights: Iterable[Float] | None, optional
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

    def _load_weights(self, weights: Iterable[Float]) -> list[float]:
        return convert_to_list(weights, float)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        if any(weight < 0 for weight in self.weights.value):
            raise ValueError("Regular objectives require non-negative weights.")

    @override
    def evaluate(self, job_completions: list[Time]) -> float:
        return sum(
            w_j
            for w_j, d_j, C_j in zip(
                self.weights.value,
                self.due_dates.value,
                job_completions,
                strict=False,
            )
            if C_j > d_j
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Σw_jU_j"

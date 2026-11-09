"""Earliness-based objectives."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import Int, JobID, Time
from cpscheduler.environment.instance import Feature, ProblemInstance
from cpscheduler.environment.objectives.base import AntiRegularObjective
from cpscheduler.environment.utils.general import convert_to_list


class TotalEarliness(AntiRegularObjective):
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
    ) -> None:
        """Initialize the Total Earliness objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: Iterable[Time] | None, optional
            The due dates for each job.
            If None is provided, the due dates will be loaded from the instance.

        """
        super().__init__()

        self.due_dates = Feature(
            name=due_dates_tag,
            preprocess=self._load_due_dates,
            shape=("n_jobs",),
            value_type="time",
        )

        if due_dates is not None:
            self.due_dates.own_data(due_dates)

    def _load_due_dates(self, due_dates: Iterable[Time]) -> list[Time]:
        return convert_to_list(due_dates, Time)

    @override
    def job_cost(self, job_id: JobID, completion: Time) -> float:
        return float(max(self.due_dates.value[job_id] - completion, 0))

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

        """
        super().__init__(due_dates_tag, due_dates)

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

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)

        if any(weight < 0 for weight in self.weights.value):
            raise ValueError(
                "Earliness objectives require non-negative weights."
            )

    @override
    def job_cost(self, job_id: JobID, completion: Time) -> float:
        return self.weights.value[job_id] * float(
            max(self.due_dates.value[job_id] - completion, 0)
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Σw_jE_j"

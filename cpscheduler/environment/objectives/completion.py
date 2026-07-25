"""Completion time related objective functions."""

from collections.abc import Iterable
from math import expm1

from typing_extensions import override

from cpscheduler.environment.constants import Float, Int, Time
from cpscheduler.environment.instance import Feature
from cpscheduler.environment.objectives.base import RegularObjective
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class TotalCompletionTime(RegularObjective):
    """Total Completion Time objective.

    This objective function aims to minimize the sum of completion times of all
    jobs, i.e., ΣC_j.
    """

    @override
    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._job_completion))

    @override
    def __call__(self, state: ScheduleState) -> float:
        return float(sum(self.completion_times(state)))

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "ΣC_j"


class WeightedCompletionTime(TotalCompletionTime):
    """Weighted Completion Time objective.

    This objective function aims to minimize the weighted sum of completion times
    of all jobs, i.e., Σw_jC_j.
    """

    weights: Feature[list[float]]

    def __init__(
        self,
        weights_tag: str = "weight",
        weights: list[Float] | None = None,
        minimize: bool = True,
    ):
        """Initialize the Weighted Completion Time objective.

        Parameters
        ----------
        weights_tag: str, optional
            The name of the job feature that contains the weights.
            Default to "weight".

        weights: list[Float] | None, optional
            The weights for each job.
            If None is provided, the weights will be loaded from the instance.
            Default to None.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.weights = Feature(
            name=weights_tag,
            preprocess=self._load_weights,
            shape=("n_jobs",),
        )

        if weights is not None:
            self.weights.own_data(weights)

    def _load_weights(self, weights: Iterable[Float]) -> list[float]:
        return convert_to_list(weights, float)

    @property
    @override
    def regular(self) -> bool:
        return all(weight >= 0 for weight in self.weights.value)

    @override
    def get_current(self, state: ScheduleState) -> float:
        weights = self.weights.value

        return sum(
            weight * float(C_j)
            for weight, C_j in zip(weights, self._job_completion, strict=False)
        )

    @override
    def __call__(self, state: ScheduleState) -> float:
        weights = self.weights.value

        return sum(
            weight * float(C_j)
            for weight, C_j in zip(
                weights, self.completion_times(state), strict=False
            )
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Σw_jC_j"


class DiscountedTotalCompletionTime(RegularObjective):
    """Discounted Total Completion Time objective.

    This objective function aims to minimize the discounted sum of completion times
    of all jobs.
    It models the case where the value of completing a job exponentially decays
    with its completion time.
    """

    discount_factor: Feature[float]

    def __init__(
        self,
        discount_factor: float = 0.99,
        minimize: bool = True,
    ):
        """Initialize the Discounted Total Completion Time objective.

        Parameters
        ----------
        discount_factor: float, optional
            The discount factor for the completion times.
            Default to 0.99.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.discount_factor = Feature(
            name="discount_factor",
            shape=(),
        )

        self.discount_factor.own_data(discount_factor)

    @override
    def get_current(self, state: ScheduleState) -> float:
        alpha = self.discount_factor.value

        return -sum(expm1(-alpha * float(C_j)) for C_j in self._job_completion)

    @override
    def __call__(self, state: ScheduleState) -> float:
        alpha = self.discount_factor.value

        return -sum(
            expm1(-alpha * float(C_j)) for C_j in self.completion_times(state)
        )

    @override
    def get_entry(self) -> str:
        if self.discount_factor.loaded:
            return f"Σ(1 - e^(-{self.discount_factor.value:.2f} C_j))"

        return "Σ(1 - e^(-r C_j))"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Σ(1 - e^(-r C_j))"


class TotalFlowTime(RegularObjective):
    """Total Flow Time objective.

    This objective function aims to minimize the total flow time of all jobs, where
    the flow time of a job is defined as its completion time minus its release time,
    i.e., F_j = C_j - r_j.
    """

    release_times: Feature[list[Time]]

    def __init__(
        self,
        release_tag: str = "release_time",
        release_times: Iterable[Int] | None = None,
        minimize: bool = True,
    ) -> None:
        """Initialize the Total Flow Time objective.

        Parameters
        ----------
        release_tag: str, optional
            The name of the job feature that contains the release times.

        release_times: Iterable[Time] | None, optional
            The release times for each job.
            If None is provided, the release times will be loaded from the instance.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.release_times = Feature(
            name=release_tag,
            preprocess=self._load_release_times,
            shape=("n_jobs",),
        )

        if release_times is not None:
            self.release_times.own_data(release_times)

    def _load_release_times(self, release_times: Iterable[Int]) -> list[Time]:
        return convert_to_list(release_times, Time)

    @override
    def get_current(self, state: ScheduleState) -> float:
        return float(
            sum(
                C_j - r_j
                for r_j, C_j in zip(
                    self.release_times.value, self._job_completion, strict=False
                )
            )
        )

    @override
    def __call__(self, state: ScheduleState) -> float:
        return float(
            sum(
                C_j - r_j
                for r_j, C_j in zip(
                    self.release_times.value,
                    self.completion_times(state),
                    strict=False,
                )
            )
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "ΣF_j"

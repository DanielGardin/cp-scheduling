"""Completion time related objective functions."""

from math import expm1

from typing_extensions import override

from cpscheduler.environment.constants import Float, Time
from cpscheduler.environment.instance import UNSET, GlobalFeature, JobFeature
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

    weights: JobFeature[float]

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

        self.weights = JobFeature(
            name=weights_tag,
            semantic="continuous",
            shape=(),
            default=(
                convert_to_list(weights, float)
                if weights is not None
                else UNSET
            ),
        )

    @property
    @override
    def regular(self) -> bool:
        return all(weight >= 0 for weight in self.weights.value)

    @override
    def get_features(self) -> list[JobFeature]:
        return [self.weights]

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

    discount_factor: GlobalFeature[float]

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

        self.discount_factor = GlobalFeature(
            name="discount_factor",
            semantic="continuous",
            shape=(),
            default=discount_factor,
        )

    @override
    def get_features(self) -> list[GlobalFeature]:
        return [self.discount_factor]

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

    release_times: JobFeature[Time]

    def __init__(
        self, release_times: str = "release_time", minimize: bool = True
    ) -> None:
        """Initialize the Total Flow Time objective.

        Parameters
        ----------
        release_times: str, optional
            The name of the job feature that contains the release times.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.release_times = JobFeature(
            name=release_times, semantic="time", shape=()
        )

    @override
    def get_features(self) -> list[JobFeature]:
        return [self.release_times]

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

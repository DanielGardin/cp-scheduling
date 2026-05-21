from math import expm1

from cpscheduler.environment.utils.general import convert_to_list
from cpscheduler.environment.constants import Time, Float
from cpscheduler.environment.instance import JobFeature, GlobalFeature, UNSET
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import RegularObjective

class TotalCompletionTime(RegularObjective):
    """
    The total completion time objective function, which aims to minimize the sum
    of completion times of all tasks.
    """

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._job_completion))

    def __call__(self, state: ScheduleState) -> float:
        return float(sum(self.completion_times(state)))

    @classmethod
    def get_general_entry(cls) -> str:
        return "ΣC_j"


class WeightedCompletionTime(TotalCompletionTime):
    """
    The weighted completion time objective function, which aims to minimize the
    weighted sum of completion times of all tasks.
    Each task has a weight associated with it, and the objective function is the
    sum of the completion times multiplied by their respective weights.
    """

    weights: JobFeature[float]

    def __init__(
        self,
        weights_tag: str = "weight",
        weights: list[Float] | None = None,
        minimize: bool = True,
    ):
        super().__init__(minimize)

        self.weights = JobFeature(
            name=weights_tag,
            elem_type=float,
            semantic="continuous",
            default=(
                convert_to_list(weights, float)
                if weights is not None else UNSET
            )
        )

    @property
    def regular(self) -> bool:
        return all(weight >= 0 for weight in self.weights.value)

    def get_features(self) -> list[JobFeature]:
        return [self.weights]

    def get_current(self, state: ScheduleState) -> float:
        weights = self.weights.value

        return sum(
            weight * float(C_j)
            for weight, C_j in zip(weights, self._job_completion)
        )

    def __call__(self, state: ScheduleState) -> float:
        weights = self.weights.value

        return sum(
            weight * float(C_j)
            for weight, C_j in zip(weights, self.completion_times(state))
        )

    @classmethod
    def get_general_entry(cls) -> str:
        return "Σw_jC_j"

class DiscountedTotalCompletionTime(RegularObjective):

    discount_factor: GlobalFeature[float]

    def __init__(
        self,
        discount_factor: float = 0.99,
        minimize: bool = True,
    ):
        super().__init__(minimize)

        self.discount_factor = GlobalFeature(
            name="discount_factor",
            pytype=float,
            semantic="continuous",
            default=discount_factor
        )

    def get_features(self) -> list[GlobalFeature]:
        return [self.discount_factor]

    def get_current(self, state: ScheduleState) -> float:
        alpha = self.discount_factor.value

        return - sum(
            expm1(-alpha * float(C_j))
            for C_j in self._job_completion
        )

    def __call__(self, state: ScheduleState) -> float:
        alpha = self.discount_factor.value

        return - sum(
            expm1(-alpha * float(C_j))
            for C_j in self.completion_times(state)
        )

    def get_entry(self) -> str:
        if self.discount_factor.loaded:
            return f"Σ(1 - e^(-{self.discount_factor.value:.2f} C_j))"

        return f"Σ(1 - e^(-r C_j))"

    @classmethod
    def get_general_entry(cls) -> str:
        return f"Σ(1 - e^(-r C_j))"

class TotalFlowTime(RegularObjective):
    """
    The total flow time objective function, which aims to minimize the sum of flow times of all
    tasks. Flow time is defined as the difference between the completion time and the release time.
    """

    release_times: JobFeature[Time]

    def __init__(
        self,
        release_times: str = "release_time",
        minimize: bool = True
    ) -> None:
        super().__init__(minimize)

        self.release_times = JobFeature(
            name=release_times,
            elem_type=Time,
            semantic="time"
        )

    def get_features(self) -> list[JobFeature]:
        return [self.release_times]

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(
            C_j - r_j
            for r_j, C_j in zip(
                self.release_times.value, self._job_completion
            )
        ))

    def __call__(self, state: ScheduleState) -> float:
        return float(sum(
            C_j - r_j
            for r_j, C_j in zip(
                self.release_times.value, self.completion_times(state)
            )
        ))

    @classmethod
    def get_general_entry(cls) -> str:
        return "ΣF_j"

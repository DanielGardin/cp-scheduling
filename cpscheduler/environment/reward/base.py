"""Base class for reward functions."""

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.objectives import Objective
from cpscheduler.environment.state import ScheduleState


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class RewardStrategy(EzPickle):
    """General reward strategy computation class."""

    def reset(self, state: ScheduleState, objective: Objective) -> None:
        """Reset the internal reward state."""

    def compute(self, state: ScheduleState, objective: Objective) -> float:
        """Compute reward, given the current state and objective values."""
        raise NotImplementedError(
            f"Reward strategy {type(self).__name__} has no implementation for "
            "the compute method."
        )


class SparseRewardStrategy(RewardStrategy):
    """Reward strategy that returns the objective value only in the terminal state."""

    @override
    def compute(self, state: ScheduleState, objective: Objective) -> float:
        reward = objective.current if state.is_terminal() else 0.0

        return -reward if objective.minimize else reward

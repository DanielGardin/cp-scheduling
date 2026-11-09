"""Base class for reward functions."""

from collections.abc import Callable

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.mixins import EzPickle
from cpscheduler.environment.objectives import Objective
from cpscheduler.environment.state import ScheduleState


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class RewardStrategy(EzPickle):
    """General reward strategy computation class."""

    def reset(
        self, state: ScheduleState, objective: Objective, minimize: bool
    ) -> None:
        """Reset the internal reward state.

        `minimize` is the sense of the environment, the objective holds none.
        """

    def compute(
        self, state: ScheduleState, objective: Objective, minimize: bool
    ) -> float:
        """Compute the reward, given the current state, objective and sense."""
        raise NotImplementedError(
            f"Reward strategy {type(self).__name__} has no implementation for "
            "the compute method."
        )


class SparseRewardStrategy(RewardStrategy):
    """Reward strategy that returns the objective value only in the terminal state.

    An infeasible terminal state receives the partial objective value.
    To penalize infeasible solutions, wrap the strategy in `InfeasibilityPenalty`.
    """

    @override
    def compute(
        self, state: ScheduleState, objective: Objective, minimize: bool
    ) -> float:
        if not state.is_terminal():
            return 0.0

        reward = objective.value
        return -reward if minimize else reward


class SparseCustomReward(RewardStrategy):
    """Reward strategy that returns a function of the objective value in the terminal state."""

    fn: Callable[[float], float]

    def __init__(self, fn: Callable[[float], float]) -> None:
        self.fn = fn

    @override
    def compute(
        self, state: ScheduleState, objective: Objective, minimize: bool
    ) -> float:
        if not state.is_terminal():
            return 0.0

        reward = self.fn(objective.value)
        return -reward if minimize else reward

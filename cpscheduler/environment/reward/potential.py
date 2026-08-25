"""Potential-difference rewards strategies."""

from typing_extensions import override

from cpscheduler.environment.objectives import Objective
from cpscheduler.environment.reward.base import RewardStrategy
from cpscheduler.environment.state import ScheduleState


class PotentialRewardStrategy(RewardStrategy):
    """Reward strategy based on the change in a state potential.

    The potential ``H(s)`` is evaluated for consecutive states and converted
    into a reward according to the optimization direction:

    - minimization:
      ``r_t = H(s_t) - gamma * H(s_{t+1})``

    - maximization:
      ``r_t = gamma * H(s_{t+1}) - H(s_t)``

    When ``gamma=1``, the undiscounted rewards telescope to the difference
    between the initial and terminal potentials. When the RL return is
    discounted by the same ``gamma``, the discounted potential differences
    telescope accordingly.

    Subclasses can override `potential` to define alternative state
    potentials, such as an objective lower bound.
    """

    _prev_potential: float
    gamma: float

    def __init__(self, gamma: float = 1.0) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        self.gamma = gamma

    def potential(self, state: ScheduleState, objective: Objective) -> float:
        """Return the potential associated with the current scheduling state.

        Parameters
        ----------
        state: ScheduleState
            Current state of the schedule.

        objective: Objective
            Objective associated with the scheduling problem.

        Returns
        -------
        The scalar potential ``H(state)`` used to compute the reward.
        """
        return objective.value

    @override
    def reset(self, state: ScheduleState, objective: Objective) -> None:
        self._prev_potential = self.potential(state, objective)

    @override
    def compute(self, state: ScheduleState, objective: Objective) -> float:
        current_potential = self.potential(state, objective)
        reward = self._prev_potential - self.gamma * current_potential

        self._prev_potential = current_potential
        return -reward if objective.minimize else reward


class LBPotentialRewardStrategy(PotentialRewardStrategy):
    """Potential-difference reward using the objective lower bound as potential."""

    @override
    def potential(self, state: ScheduleState, objective: Objective) -> float:
        return objective.lb


class UBPotentialRewardStrategy(PotentialRewardStrategy):
    """Potential-difference reward using the objective upper bound as potential."""

    @override
    def potential(self, state: ScheduleState, objective: Objective) -> float:
        return objective.ub

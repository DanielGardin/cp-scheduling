"""Penalties for episodes that end in an infeasible state.

An infeasible state is not a schedule, so its objective value says nothing
about how bad failing is.
This is particularly important for regular objectives, whose partial value
is always at least as good as any compact completion of the partial schedule.
So an agent rewarded by the partial value in a failing state learns to fail as
early as possible.
"""

from collections.abc import Callable

from typing_extensions import override

from cpscheduler.environment.constants import MAX_TIME, Time
from cpscheduler.environment.objectives import ComposedObjective, Objective
from cpscheduler.environment.objectives.base import (
    CompletionObjective,
    completion_times,
)
from cpscheduler.environment.reward.base import RewardStrategy
from cpscheduler.environment.state import ScheduleState

PenaltyFn = Callable[[ScheduleState, Objective, bool], float]
"A penalty as a function of the failing state, the objective and the sense."


def _pessimistic_horizon(state: ScheduleState) -> Time:
    """Return a time no compact complete schedule would finish after."""
    horizon: int = max(state.get_latest_end(), 0)

    for task_id in range(state.n_tasks):
        if not state.is_fixed(task_id) and state.is_feasible(task_id):
            start_lb = state.get_start_lb(task_id)
            if horizon < start_lb < MAX_TIME:
                horizon = start_lb

    processing_times = state.instance.processing_times
    for task_id in range(state.n_tasks):
        horizon += max(
            (
                processing_times[task_id][machine_id]
                for machine_id in state.get_original_machines(task_id)
            ),
            default=0,
        )

    return Time(min(horizon, MAX_TIME))


def _pessimistic_value(state: ScheduleState, objective: Objective) -> float:
    """Return the value of finishing every incomplete job at the horizon.

    Only regular objectives get worse when work is left, any other objective
    is scored by its value.
    """
    if isinstance(objective, ComposedObjective):
        return sum(
            coef
            * (_pessimistic_value(state, term) if coef >= 0 else term.value)
            for term, coef in zip(
                objective.objectives, objective.coefficients, strict=True
            )
        )

    if not isinstance(objective, CompletionObjective) or not objective.regular:
        return objective.value

    horizon = _pessimistic_horizon(state)
    completions = completion_times(state)

    for task_id in range(state.n_tasks):
        if not state.is_fixed(task_id):
            completions[state.get_job_id(task_id)] = horizon

    return objective.evaluate(completions)


def _pessimistic_gap(
    state: ScheduleState, objective: Objective, minimize: bool
) -> float:
    """Return how much worse finishing at the horizon is than the partial value."""
    if not minimize:
        return 0.0

    return max(_pessimistic_value(state, objective) - objective.value, 0.0)


# FUTURE: The default penalty is correct, failing never returns more than
# finishing, but it is measured in objective units. Its size follows the size of
# the instance, so an agent trained over many instances sees failures weigh
# differently from one instance to the next. Two alternatives trade this off:
#
# - Normalized returns: scale the rewards by a range fixed at reset, from the
#   optimistic bound to a pessimistic value, so that failure gets a constant
#   return, such as -1. Every objective must then provide that range.
# - Learned penalty: constrained-MDP methods (e.g. RCPO, Tessler et al. 2019)
#   adjust the weight of the penalty during training, which needs support from
#   the training loop rather than from the environment.
#
# Either one only matters for problems whose masks cannot prevent dead ends,
# such as deadlines, horizons, no-wait links, alternative groups and
# non-renewable resources. Jm||C_max, for instance, never fails.
class InfeasibilityPenalty(RewardStrategy):
    """Subtract a penalty from the reward of the step that makes the state infeasible.

    The wrapped strategy computes the reward as usual, but when the state becomes
    infeasible, a penalty is subtracted from that step's reward.
    Episodes end at the first infeasible state, so the penalty is applied once.

    By default, a minimized objective is penalized by the gap between its
    partial value and the value of finishing every incomplete job at a
    pessimistic horizon, so failing scores like the worst completion.

    Parameters
    ----------
    strategy: RewardStrategy
        The reward strategy to wrap.

    penalty: float | Callable[[ScheduleState, Objective, bool], float] | None
        A constant, or a function of the failing state, the objective and
        whether the environment minimizes it.
        Defaults to the pessimistic gap.

    Notes
    -----
    The default penalty is instance-specific, working as a pessimistic bound on
    the objective value when the objective is regular and minimized.
    Because the penalty varies with the instance, it may be hard to generalize
    across instances when compared to a constant penalty.
    """

    strategy: RewardStrategy
    penalty_fn: PenaltyFn | None
    penalty: float

    def __init__(
        self,
        strategy: RewardStrategy,
        penalty: float | PenaltyFn | None = None,
    ) -> None:
        self.strategy = strategy
        self.penalty = 0.0
        self.penalty_fn = None

        if penalty is None:
            self.penalty_fn = _pessimistic_gap

        elif callable(penalty):
            self.penalty_fn = penalty

        else:
            self.penalty = float(penalty)

    @override
    def reset(
        self, state: ScheduleState, objective: Objective, minimize: bool
    ) -> None:
        self.strategy.reset(state, objective, minimize)

    @override
    def compute(
        self, state: ScheduleState, objective: Objective, minimize: bool
    ) -> float:
        reward = self.strategy.compute(state, objective, minimize)

        if not state.infeasible:
            return reward

        if self.penalty_fn is not None:
            return reward - self.penalty_fn(state, objective, minimize)

        return reward - self.penalty

    def __repr__(self) -> str:
        """Return a string representation of the wrapper."""
        penalty = (
            getattr(self.penalty_fn, "__name__", repr(self.penalty_fn))
            if self.penalty_fn is not None
            else repr(self.penalty)
        )

        return f"{type(self).__name__}({self.strategy!r}, penalty={penalty})"

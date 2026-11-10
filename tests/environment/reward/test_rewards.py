from collections.abc import Callable

import pytest

from cpscheduler.environment import (
    ComposedObjective,
    DeadlineConstraint,
    HorizonConstraint,
    Makespan,
    Objective,
    SchedulingEnv,
    SingleMachineSetup,
    TotalCompletionTime,
)
from cpscheduler.environment.reward import (
    InfeasibilityPenalty,
    LBPotentialRewardStrategy,
    PotentialRewardStrategy,
    RewardStrategy,
    SparseRewardStrategy,
)
from cpscheduler.environment.state import ScheduleState


def _deadline_env(
    objective: Objective, reward: RewardStrategy | None = None
) -> SchedulingEnv:
    # Running task 0 first makes task 1 miss its deadline
    env = SchedulingEnv(
        SingleMachineSetup(),
        [DeadlineConstraint(due_dates=[10, 3])],
        objective,
        reward=reward,
        instance={"processing_time": [2, 2]},
    )
    env.reset()

    return env


def _failing_return(reward: RewardStrategy | None) -> float:
    env = _deadline_env(Makespan(), reward)

    _, step_return, _, truncated, _ = env.step(("execute", 0))

    assert truncated

    return float(step_return)


@pytest.mark.parametrize(
    "strategy",
    [SparseRewardStrategy, PotentialRewardStrategy, LBPotentialRewardStrategy],
)
def test_failing_returns_less_than_finishing(
    strategy: Callable[[], RewardStrategy],
) -> None:
    # Otherwise maximizing the return teaches agents to fail as early as possible
    complete = _deadline_env(Makespan(), InfeasibilityPenalty(strategy()))

    complete_return = 0.0
    for task in (1, 0):
        _, step_return, _, truncated, _ = complete.step(("execute", task))
        complete_return += step_return

        assert not truncated

    assert _failing_return(InfeasibilityPenalty(strategy())) < complete_return


def _partial_value(
    state: ScheduleState, objective: Objective, minimize: bool
) -> float:
    return objective.value


@pytest.mark.parametrize(
    ("reward", "expected"),
    [
        # The partial makespan is 2, and the pessimistic horizon 2 + 2 + 2
        pytest.param(SparseRewardStrategy(), -2, id="unwrapped"),
        pytest.param(None, -6, id="default"),
        pytest.param(
            InfeasibilityPenalty(SparseRewardStrategy()), -6, id="pessimistic"
        ),
        pytest.param(
            InfeasibilityPenalty(SparseRewardStrategy(), 100),
            -102,
            id="constant",
        ),
        pytest.param(
            InfeasibilityPenalty(SparseRewardStrategy(), _partial_value),
            -4,
            id="function",
        ),
    ],
)
def test_only_the_wrapper_penalizes_failure(
    reward: RewardStrategy | None, expected: float
) -> None:
    assert _failing_return(reward) == expected


def test_feasible_episodes_are_not_penalized() -> None:
    returns = []
    for reward in (
        SparseRewardStrategy(),
        InfeasibilityPenalty(SparseRewardStrategy(), 100),
    ):
        env = _deadline_env(Makespan(), reward)

        _, step_return, _, truncated, _ = env.step(
            [("execute", 1), ("execute", 0)]
        )

        assert not truncated

        returns.append(step_return)

    assert returns == [-4, -4]


def test_composed_failure_is_worse_than_its_only_schedule() -> None:
    env = _deadline_env(ComposedObjective([Makespan(), TotalCompletionTime()]))

    _, reward, _, truncated, _ = env.step(("execute", 0))

    # The only complete schedule runs 1 then 0, with C_max = 4 and ΣC_j = 2 + 4
    assert truncated
    assert reward < -(4 + 6)


@pytest.mark.parametrize(
    ("constraints", "expected"),
    [
        pytest.param([], (1.0, True, False), id="success"),
        pytest.param([HorizonConstraint(3)], (0.0, True, True), id="failure"),
    ],
)
def test_default_objective_rewards_success(
    constraints: list[HorizonConstraint], expected: tuple[float, bool, bool]
) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(), constraints, instance={"processing_time": [2, 2]}
    )
    env.reset()

    _, reward, terminated, truncated, _ = env.step(
        [("execute", 0), ("execute", 1)]
    )

    assert (reward, terminated, truncated) == expected

from collections import Counter
from math import exp
from typing import TYPE_CHECKING, Literal

import pytest

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    Makespan,
    SchedulingEnv,
)
from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.heuristics.pdrs import PriorityDispatchingRule
from cpscheduler.heuristics.pdrs.base import solve_p_star_temperature

if TYPE_CHECKING:
    from cpscheduler.environment.backend.actions import SingleAction


class _Fixed(PriorityDispatchingRule):
    def __init__(self, priorities: list[float], seed: int = 0) -> None:
        super().__init__(seed)
        self.priorities = priorities

    def priority_score(self, obs: DefaultObservation) -> list[float]:
        return self.priorities


def _observation(n_tasks: int) -> DefaultObservation:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(n_tasks),
        objective=Makespan(),
        observation=DefaultObservation(),
        instance={"processing_time": [1] * n_tasks},
    )
    obs, _ = env.reset()

    return obs


@pytest.mark.heuristics
@pytest.mark.parametrize(
    ("method", "first"),
    [("sample", ("execute", 0)), ("sample_ranking", ("submit", 0))],
)
def test_first_choice_follows_the_softmax_of_the_priorities(
    method: Literal["sample", "sample_ranking"], first: "SingleAction"
) -> None:
    # The first choice of a Gumbel-perturbed ranking is a Plackett-Luce draw
    rule = _Fixed([2.0, 0.0, 0.0])
    obs = _observation(3)
    n_samples = 40_000

    choices: Counter[SingleAction | None] = Counter(
        rule.sample(obs)
        if method == "sample"
        else rule.sample_ranking(obs, "parallel")[0]
        for _ in range(n_samples)
    )

    assert choices[first] / n_samples == pytest.approx(
        exp(2) / (exp(2) + 2), abs=0.01
    )


def test_temperature_only_depends_on_priority_differences() -> None:
    base = solve_p_star_temperature([3.0, 2.0, 1.0], 0.6, 5)

    assert base > 0
    assert solve_p_star_temperature([103.0, 102.0, 101.0], 0.6, 5) == (
        pytest.approx(base)
    )
    assert solve_p_star_temperature([-1.0, -2.0, -3.0], 0.6, 5) == (
        pytest.approx(base)
    )


@pytest.mark.heuristics
def test_temperature_reaches_the_target_probability() -> None:
    rule = _Fixed([-1.0, -2.0, -3.0, -4.0])
    obs = _observation(4)
    n_samples = 30_000

    hits = sum(
        rule.sample(obs, target_prob=0.6, n_iter=30) == ("execute", 0)
        for _ in range(n_samples)
    )

    assert hits / n_samples == pytest.approx(0.6, abs=0.02)


def test_single_option_is_always_sampled() -> None:
    rule = _Fixed([-1.0])

    assert rule.sample(_observation(1), target_prob=0.9) == ("execute", 0)

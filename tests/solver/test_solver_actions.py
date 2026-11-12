import pytest

pytest.importorskip("pyomo")
from cpscheduler.environment import (
    Makespan,
    SchedulingEnv,
    SingleMachineSetup,
)
from cpscheduler.solver import SchedulingSolver
from cpscheduler.solver.naive.bnb_naive import NaiveBnBFormulation
from tests.solver_checks import replay


def test_solver_actions_replay_on_a_transition_without_clock() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        transition="tetris",
        instance={"processing_time": [1, 2]},
    )
    env.reset()

    action, value, _ = SchedulingSolver(env, NaiveBnBFormulation()).solve(
        quiet=True
    )

    assert replay(env, action) == value

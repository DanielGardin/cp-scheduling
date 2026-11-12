from random import Random

import pytest

pytest.importorskip("pyomo")
from cpscheduler.environment import (
    JobShopSetup,
    Makespan,
    SchedulingEnv,
    SingleMachineSetup,
    TotalCompletionTime,
    UnrelatedParallelMachineSetup,
)
from cpscheduler.solver import SchedulingSolver
from cpscheduler.solver.naive.bnb_naive import NaiveBnBFormulation
from tests.solver_checks import (
    REGULAR_OBJECTIVES,
    InstanceData,
    brute_force,
    regular_instance,
    replay,
)


@pytest.mark.solver
@pytest.mark.parametrize("objective", REGULAR_OBJECTIVES)
def test_bnb_is_optimal_on_a_single_machine(objective: str) -> None:
    factory, features = REGULAR_OBJECTIVES[objective]
    rng = Random(3)

    for _ in range(5):
        instance = regular_instance(rng, 5, features)

        def make(instance: InstanceData = instance) -> SchedulingEnv:
            return SchedulingEnv(
                SingleMachineSetup(),
                objective=factory(),
                transition="tetris",
                instance=instance,
            )

        env = make()
        env.reset()

        action, value, result = SchedulingSolver(
            env, NaiveBnBFormulation()
        ).solve(quiet=True)

        assert result.optimal
        assert value == pytest.approx(brute_force(make, 5))
        assert replay(make(), action) == pytest.approx(value)


@pytest.mark.solver
@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("transition", ["des", "tetris"])
def test_bnb_reports_the_value_of_the_schedule_it_returns(
    transition: str, seed: int
) -> None:
    # Machines selected during the search and in the final replay must agree
    rng = Random(seed)
    processing = [[rng.randint(1, 4) for _ in range(3)] for _ in range(5)]

    env = SchedulingEnv(
        UnrelatedParallelMachineSetup(),
        objective=TotalCompletionTime(),
        transition=transition,
        instance={"processing_time": processing},
    )
    env.reset()

    formulation = NaiveBnBFormulation()
    _, value, _ = SchedulingSolver(env, formulation).solve(
        quiet=True, time_limit=5
    )

    assert value == sum(
        start + processing[task][machine]
        for task, (machine, start) in formulation.solution.items()
    )


def test_warm_start_at_the_optimum_is_returned() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=TotalCompletionTime(),
        transition="tetris",
        instance={"processing_time": [1, 2]},
    )
    env.reset()

    solver = SchedulingSolver(env, NaiveBnBFormulation())
    solver.warm_start([("execute", 0), ("execute", 1)])
    action, value, _ = solver.solve(quiet=True)

    assert value == 4
    assert replay(env, action) == 4


def test_problem_decided_by_forced_actions_has_one_solution() -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        instance={
            "job": [0, 0],
            "operation": [0, 1],
            "machine": [0, 1],
            "processing_time": [1, 2],
        },
    )
    env.reset()

    action, value, result = SchedulingSolver(env, NaiveBnBFormulation()).solve(
        quiet=True
    )

    assert (value, result.solutions) == (3, 1)
    assert replay(env, action) == 3


def test_bnb_rejects_maximization() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        sense="maximize",
        instance={"processing_time": [1, 2]},
    )
    env.reset()

    with pytest.raises(ValueError, match="minimization"):
        SchedulingSolver(env, NaiveBnBFormulation())

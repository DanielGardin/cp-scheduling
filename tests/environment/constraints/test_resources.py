from random import Random

import pytest

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    Makespan,
    NonRenewableResourceConstraint,
    ResourceConstraint,
    SchedulingEnv,
    SingleMachineSetup,
)
from tests.schedule_checks import (
    capacity_violations,
    run_random_episode,
    schedule_of,
)


@pytest.mark.env
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("transition", ["des", "tetris"])
def test_renewable_resource_capacity_is_respected(
    transition: str, seed: int
) -> None:
    rng = Random(seed)
    capacity = rng.randint(3, 6)
    usage = [rng.randint(1, capacity) for _ in range(7)]

    env = SchedulingEnv(
        IdenticalParallelMachineSetup(7),
        [ResourceConstraint(capacity=capacity, resources=usage)],
        Makespan(),
        transition=transition,
        instance={"processing_time": [rng.randint(1, 6) for _ in range(7)]},
    )
    run_random_episode(env, rng)

    assert not capacity_violations(schedule_of(env.state), usage, capacity)


def test_task_using_more_than_the_stock_fails_on_reset() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [NonRenewableResourceConstraint(capacity=1, resources=[5])],
        Makespan(),
        instance={"processing_time": [1]},
    )

    with pytest.raises(RuntimeError, match="stale state"):
        env.reset()


def test_non_renewable_resource_is_consumed() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [NonRenewableResourceConstraint(capacity=2, resources=[2, 1])],
        Makespan(),
        instance={"processing_time": [1, 1]},
    )
    env.reset()

    *_, truncated, _ = env.step(("execute", 0))

    assert truncated
    assert env.state.infeasible

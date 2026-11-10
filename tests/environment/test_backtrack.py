from collections.abc import Callable
from copy import deepcopy
from random import Random
from typing import NamedTuple, TypeAlias

import pytest

from cpscheduler.environment import (
    DeadlineConstraint,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    Makespan,
    OptionalityConstraint,
    RejectableConstraint,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
    TotalCompletionTime,
    UnrelatedParallelMachineSetup,
)
from cpscheduler.environment.state.csp import TaskDomains
from cpscheduler.environment.utils.protocols import InfoType
from tests.schedule_checks import (
    BACKENDS,
    Schedule,
    random_job_shop,
    schedule_of,
)

N_EPISODES = 25
N_DIVES = 4

Snapshot: TypeAlias = tuple[
    TaskDomains, int, bool, float, float, float, InfoType, int, list[int]
]
EnvFactory: TypeAlias = Callable[[Random, str], SchedulingEnv]


class Coverage(NamedTuple):
    verified: int
    from_terminal: int
    from_failure: int


def _snapshot(env: SchedulingEnv) -> Snapshot:
    return (
        deepcopy(env.state.domains),
        env.state.remaining_tasks,
        env.state.infeasible,
        env.objective.value,
        env.objective.lb,
        env.objective.ub,
        env.backend.get_info(),
        env.event_count,
        sorted(env.get_action_support()),
    )


def _random_parallel(rng: Random, n_tasks: int) -> dict[str, list[int]]:
    processing = [rng.randint(1, 9) for _ in range(n_tasks)]
    release = [rng.randint(0, 9) for _ in range(n_tasks)]

    # Every task can meet its deadline alone, but not in every order
    return {
        "processing_time": processing,
        "release_time": release,
        "due_time": [
            r + p + rng.randint(0, 8)
            for r, p in zip(release, processing, strict=True)
        ],
    }


FACTORIES: dict[str, EnvFactory] = {
    "job_shop": lambda rng, backend: SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        backend=backend,
        instance=random_job_shop(rng, 3, 3),
    ),
    "total_completion": lambda rng, backend: SchedulingEnv(
        JobShopSetup(),
        objective=TotalCompletionTime(),
        backend=backend,
        instance=random_job_shop(rng, 3, 3),
    ),
    "parallel_released": lambda rng, backend: SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [ReleaseDateConstraint()],
        Makespan(),
        backend=backend,
        instance={
            key: values
            for key, values in _random_parallel(rng, 6).items()
            if key != "due_time"
        },
    ),
    # Random dives often miss a deadline or reject a required task, while
    # optional tasks that cannot meet theirs become absent instead
    "deadlines": lambda rng, backend: SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [ReleaseDateConstraint(), DeadlineConstraint()],
        TotalCompletionTime(),
        backend=backend,
        instance=_random_parallel(rng, 6),
    ),
    "rejectable_jobs": lambda rng, backend: SchedulingEnv(
        JobShopSetup(),
        [RejectableConstraint()],
        Makespan(),
        backend=backend,
        instance=random_job_shop(rng, 3, 3),
    ),
    "optional": lambda rng, backend: SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [
            ReleaseDateConstraint(),
            DeadlineConstraint(),
            OptionalityConstraint(),
        ],
        TotalCompletionTime(),
        backend=backend,
        instance=_random_parallel(rng, 6),
    ),
}


def _dive(
    env: SchedulingEnv, rng: Random, checkpoints: list[tuple[int, Snapshot]]
) -> None:
    while not env.state.is_terminal():
        if rng.random() < 0.4:
            checkpoints.append((env.checkpoint(), _snapshot(env)))

        support = env.get_action_support()
        kind = "reject" if rng.random() < 0.1 else "execute"
        env.step((kind, rng.choice(support)) if support else None)

    # Searches also checkpoint complete schedules before branching again
    if not env.state.infeasible and rng.random() < 0.3:
        checkpoints.append((env.checkpoint(), _snapshot(env)))


def _explore(env: SchedulingEnv, rng: Random) -> Coverage:
    env.reset()
    checkpoints = [(env.checkpoint(), _snapshot(env))]
    verified = from_terminal = from_failure = 0

    for dive in range(N_DIVES):
        _dive(env, rng, checkpoints)

        from_terminal += 1
        from_failure += env.state.infeasible

        last_dive = dive == N_DIVES - 1
        n_unwind = (
            len(checkpoints) if last_dive else rng.randint(1, len(checkpoints))
        )

        for _ in range(n_unwind):
            mark, expected = checkpoints.pop()
            env.backtrack(mark)
            verified += 1

            assert _snapshot(env) == expected

        # The root is restored to dive again
        if not checkpoints:
            checkpoints.append((env.checkpoint(), _snapshot(env)))

    return Coverage(verified, from_terminal, from_failure)


@pytest.mark.env
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("factory", FACTORIES)
def test_backtrack_restores_exactly_every_checkpoint(
    factory: str, backend: str
) -> None:
    rng = Random(7)

    coverage = [
        _explore(FACTORIES[factory](rng, backend), rng)
        for _ in range(N_EPISODES)
    ]

    assert sum(c.from_terminal for c in coverage) == N_EPISODES * N_DIVES
    assert sum(c.verified for c in coverage) >= N_EPISODES * N_DIVES

    if factory == "deadlines":
        assert sum(c.from_failure for c in coverage) > 0


def test_backtracked_queue_does_not_leak_into_new_decisions() -> None:
    # Executing a late task defers the queue, which must be undone as well
    env = SchedulingEnv(
        SingleMachineSetup(),
        [ReleaseDateConstraint()],
        Makespan(),
        instance={"processing_time": [1, 1], "release_time": [0, 50]},
    )
    env.reset()

    mark = env.checkpoint()
    env.step(("execute", 1))
    env.backtrack(mark)
    env.step(("execute", 0))

    assert env.state.get_start(0) == 0


@pytest.mark.parametrize("backend", ["des", "tetris"])
def test_machine_ties_do_not_depend_on_explored_branches(backend: str) -> None:
    def play(explore: bool) -> Schedule:
        env = SchedulingEnv(
            UnrelatedParallelMachineSetup(),
            objective=Makespan(),
            backend=backend,
            instance={"processing_time": [[2, 2, 2]] * 5},
        )
        env.reset()

        if explore:
            mark = env.checkpoint()
            env.step(("execute", 0, 2))
            env.backtrack(mark)

        env.step([("execute", task) for task in range(5)])

        return schedule_of(env.state)

    assert play(explore=True) == play(explore=False)


def test_state_equality_ignores_the_order_of_feasible_machines() -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(3),
        objective=Makespan(),
        backend="tetris",
        instance={"processing_time": [1, 1]},
    )
    env.reset()

    state = deepcopy(env.state)
    mark = env.checkpoint()
    env.step(("execute", 0, 2))
    env.backtrack(mark)

    assert env.state == state

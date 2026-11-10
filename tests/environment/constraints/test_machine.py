from random import Random
from typing import TYPE_CHECKING

import pytest

from cpscheduler.environment import (
    BatchConstraint,
    IdenticalParallelMachineSetup,
    MachineBreakdownConstraint,
    Makespan,
    SchedulingEnv,
    SingleMachineSetup,
)
from tests.schedule_checks import (
    calendar_violations,
    max_concurrency,
    run_random_episode,
    schedule_of,
)

if TYPE_CHECKING:
    from cpscheduler.environment.constants import Int


@pytest.mark.parametrize("backend", ["des", "tetris"])
def test_task_does_not_cross_a_breakdown(backend: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [MachineBreakdownConstraint({0: [(2, 4)]})],
        Makespan(),
        backend=backend,
        instance={"processing_time": [3, 1]},
    )
    env.reset()

    # Task 1 fits before the breakdown, task 0 in [1, 4) would cross it
    env.step([("execute", 1), ("execute", 0)])

    assert (env.state.get_start(1), env.state.get_end(1)) == (0, 1)
    assert (env.state.get_start(0), env.state.get_end(0)) == (4, 7)


def test_breakdowns_from_instance_data() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [MachineBreakdownConstraint()],
        Makespan(),
        instance={"processing_time": [3], "breakdown": [[(0, 4)]]},
    )
    env.reset()

    env.step(("execute", 0))

    assert env.state.get_start(0) == 4


def test_every_breakdown_of_a_calendar_is_respected() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [MachineBreakdownConstraint({0: [(0, 2), (4, 6)]})],
        Makespan(),
        instance={"processing_time": [2, 2, 2]},
    )
    env.reset()

    env.step([("submit", 0), ("submit", 1), ("submit", 2)])

    assert sorted(env.state.get_start(t) for t in range(3)) == [2, 6, 8]


@pytest.mark.env
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("backend", ["des", "step", "tetris"])
def test_no_schedule_overlaps_a_breakdown(backend: str, seed: int) -> None:
    # Idle machines see their clock advanced by the others
    rng = Random(seed)
    calendar: dict[Int, list[tuple[Int, Int]]] = {
        0: [(3, 6), (9, 12)],
        1: [(1, 2)],
    }

    env = SchedulingEnv(
        IdenticalParallelMachineSetup(3),
        [MachineBreakdownConstraint(calendar)],
        Makespan(),
        backend=backend,
        instance={"processing_time": [rng.randint(1, 5) for _ in range(8)]},
    )
    run_random_episode(env, rng)

    assert not env.state.infeasible
    assert not calendar_violations(schedule_of(env.state), calendar)


def test_permanent_breakdown_removes_machines_too_short_for_a_task() -> None:
    # Machine 1 goes down at t=3 forever, a task of length 5 only fits on 0
    constraint = MachineBreakdownConstraint.from_machine_step_function(
        {0: 2, 3: 1}
    )
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [constraint],
        Makespan(),
        instance={"processing_time": [5]},
    )
    env.reset()

    assert env.state.get_machines(0) == [0]


@pytest.mark.parametrize("backend", ["des", "tetris"])
@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_batch_capacity_is_reached_and_not_exceeded(
    backend: str, capacity: int
) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        [BatchConstraint(capacity)],
        Makespan(),
        backend=backend,
        instance={"processing_time": [3] * 7},
    )
    run_random_episode(env, Random(capacity))

    schedule = schedule_of(env.state)

    assert len(schedule) == 7
    assert max_concurrency(schedule.values()) == capacity


def test_added_breakdowns_hold_in_every_episode() -> None:
    constraint = MachineBreakdownConstraint()
    constraint.add_breakdown(0, 0, 4)

    env = SchedulingEnv(
        SingleMachineSetup(),
        [constraint],
        Makespan(),
        instance={"processing_time": [3]},
    )

    starts = []
    for _ in range(2):
        env.reset()
        env.step(("execute", 0))
        starts.append(env.state.get_start(0))

    assert starts == [4, 4]


def test_capacities_set_by_machine_hold_in_every_episode() -> None:
    constraint = BatchConstraint()
    constraint.set_capacity(0, 2)

    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        [constraint],
        Makespan(),
        instance={"processing_time": [3] * 3},
    )

    concurrency = []
    for _ in range(2):
        env.reset()
        env.step([("execute", t) for t in range(3)])
        concurrency.append(max_concurrency(schedule_of(env.state).values()))

    assert concurrency == [2, 2]

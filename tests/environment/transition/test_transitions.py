from collections.abc import Callable
from random import Random
from typing import TypeAlias

import pytest

from cpscheduler.environment import (
    DefaultObservation,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    Makespan,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
    UnrelatedParallelMachineSetup,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.transition import (
    Clock,
    ConflictSetCandidates,
    Instruction,
    Transition,
    register_instruction,
)
from cpscheduler.environment.transition.presets import delta_gt_tetris
from cpscheduler.heuristics.pdrs import (
    MostOperationsRemaining,
    MostWorkRemaining,
    PriorityDispatchingRule,
    ShortestProcessingTime,
)
from tests.schedule_checks import (
    TRANSITIONS,
    JobShop,
    Starts,
    is_active,
    is_non_delay,
    random_job_shop,
    reachable_schedules,
    semi_active_schedules,
)

GENERAL_PRESETS = ("tetris", "gt_tetris", "nd_tetris")


class ExecuteOnLastMachine(Instruction[Clock]):
    task_id: int

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id

    def process(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> None:
        machine = max(state.get_machines(self.task_id))
        machine, start = transition.place(state, self.task_id, machine)

        transition.placement.apply(state, self.task_id, machine, start)


register_instruction(ExecuteOnLastMachine, "execute_last", Clock)

EnvFactory: TypeAlias = Callable[[Random, str], SchedulingEnv]
ScheduleClass: TypeAlias = Callable[[Starts, JobShop], bool]


def _semi_active(starts: Starts, instance: JobShop) -> bool:
    return True


FACTORIES: dict[str, EnvFactory] = {
    "job_shop": lambda rng, transition: SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance=random_job_shop(rng, 4, 3),
    ),
    "identical_released": lambda rng, transition: SchedulingEnv(
        IdenticalParallelMachineSetup(3),
        [ReleaseDateConstraint()],
        Makespan(),
        transition=transition,
        instance={
            "processing_time": [rng.randint(1, 9) for _ in range(8)],
            "release_time": [rng.randint(0, 10) for _ in range(8)],
        },
    ),
    "unrelated": lambda rng, transition: SchedulingEnv(
        UnrelatedParallelMachineSetup(),
        objective=Makespan(),
        transition=transition,
        instance={
            "processing_time": [
                [rng.randint(1, 9) for _ in range(3)] for _ in range(7)
            ]
        },
    ),
}

# The DES only executes at its current time, so it is the non-delay anchor
# for the presets that claim the same class.
CLASSES: dict[str, ScheduleClass] = {
    "tetris": _semi_active,
    "gt_tetris": is_active,
    "nd_tetris": is_non_delay,
    "des": is_non_delay,
}

RULES: dict[str, type[PriorityDispatchingRule]] = {
    "spt": ShortestProcessingTime,
    "mor": MostOperationsRemaining,
    "mwr": MostWorkRemaining,
}


def _delta_gt_env(instance: JobShop, delta: float) -> SchedulingEnv:
    return SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=delta_gt_tetris(delta),
        instance=instance,
    )


def _dispatched_starts(instance: JobShop, transition: str, rule: str) -> Starts:
    env: SchedulingEnv[DefaultObservation] = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance=instance,
    )
    agent = RULES[rule]()

    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, done, *_ = env.step(agent(obs))

    return tuple(env.state.get_start(task) for task in range(env.state.n_tasks))


@pytest.mark.env
@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("transition", GENERAL_PRESETS)
def test_executed_tasks_come_from_the_candidate_set(
    transition: str, factory: str
) -> None:
    for seed in range(10):
        rng = Random(seed)
        env = FACTORIES[factory](rng, transition)
        env.reset()

        while not env.state.is_terminal():
            mask = env.transition.get_action_mask(env.state)
            task, machine = rng.choice(
                sorted((t, m) for m, tasks in mask.items() for t in tasks)
            )

            # Without a machine, the placement chooses it
            env.step(
                ("execute", task, machine)
                if rng.random() < 0.5
                else ("execute", task)
            )

            assert task in mask[env.state.get_assignment(task)]


@pytest.mark.parametrize("transition", GENERAL_PRESETS)
def test_executing_a_locked_task_is_an_error(transition: str) -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance={
            "job": [0, 0, 1, 1],
            "operation": [0, 1, 0, 1],
            "machine": [1, 0, 0, 1],
            "processing_time": [2, 1, 2, 1],
        },
    )
    env.reset()

    with pytest.raises(RuntimeError, match="task_id=1"):
        env.step(("execute", 1, 0))


def test_gt_rejects_a_task_outside_the_conflict_set() -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition="gt_tetris",
        instance={
            "job": [0, 0, 1, 1],
            "operation": [0, 1, 0, 1],
            "machine": [1, 0, 0, 1],
            "processing_time": [2, 1, 2, 1],
        },
    )
    env.reset()

    with pytest.raises(RuntimeError, match="task_id=2, machine_id=0"):
        env.step(("execute", 2, 0))


@pytest.mark.parametrize(("first_machine", "critical"), [(1, 1), (0, 0)])
def test_gt_critical_machine_ties_go_to_the_first_unlocked_task(
    first_machine: int, critical: int
) -> None:
    # Both first operations end at 2, the tie is broken by task order only
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition="gt_tetris",
        instance={
            "job": [0, 0, 1, 1],
            "operation": [0, 1, 0, 1],
            "machine": [first_machine, 1 - first_machine] * 2,
            "processing_time": [2, 1, 2, 1],
        },
    )
    env.reset()

    assert list(env.transition.get_action_mask(env.state)) == [critical]


def test_gt_conflict_set_is_on_the_critical_machine() -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        objective=Makespan(),
        transition="gt_tetris",
        instance={"processing_time": [3, 3, 3]},
    )
    env.reset()

    env.step(("execute", 0))

    assert list(env.transition.get_action_mask(env.state)) == [1]

    env.step(("execute", 1))

    assert env.state.get_assignment(1) == 1


@pytest.mark.env
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("transition", CLASSES)
def test_transition_reaches_exactly_its_schedule_class(
    transition: str, seed: int
) -> None:
    instance = random_job_shop(Random(seed), 3, 3)
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance=instance,
    )

    expected = {
        starts
        for starts in semi_active_schedules(instance)
        if CLASSES[transition](starts, instance)
    }

    assert reachable_schedules(env) == expected


@pytest.mark.parametrize(
    ("transition", "starts"),
    [("tetris", [0, 2, 3, 5]), ("gt_tetris", [0, 2, 0, 2])],
)
def test_submitted_tasks_decode_a_priority_list(
    transition: str, starts: list[int]
) -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance={
            "job": [0, 0, 1, 1],
            "operation": [0, 1, 0, 1],
            "machine": [1, 0, 0, 1],
            "processing_time": [2, 1, 2, 1],
        },
    )
    env.reset()

    # Each job is submitted backwards, so a task waits for its predecessor
    env.step([("submit", 1), ("submit", 0), ("submit", 3), ("submit", 2)])

    assert [env.state.get_start(task) for task in range(4)] == starts
    assert env.transition.is_empty()


@pytest.mark.parametrize("transition", GENERAL_PRESETS)
def test_waiting_is_refused_without_a_clock(transition: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        transition=transition,
        instance={"processing_time": [1]},
    )
    env.reset()

    with pytest.raises(ValueError, match="GlobalClock"):
        env.step(("noop",))


@pytest.mark.parametrize("transition", ["des", "step", *GENERAL_PRESETS])
def test_blocking_execute_of_a_locked_task_names_the_lock(
    transition: str,
) -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance={
            "job": [0, 0],
            "operation": [0, 1],
            "machine": [0, 1],
            "processing_time": [2, 3],
        },
    )
    env.reset()

    with pytest.raises(RuntimeError, match="is locked"):
        env.step([("execute", 1), ("execute", 0)])


@pytest.mark.parametrize("transition", ["des", *GENERAL_PRESETS])
def test_halt_leaves_the_next_instructions_queued(transition: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        transition=transition,
        instance={"processing_time": [1, 1]},
    )
    env.reset()

    env.step([("execute", 0), ("halt",), ("execute", 1)])

    assert env.state.get_assigned_tasks() == [0]
    assert not env.transition.is_empty()

    env.step()

    assert env.state.is_terminal()


def test_waiting_on_the_unit_clock_jumps_to_the_next_release() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [ReleaseDateConstraint()],
        Makespan(),
        transition="step",
        instance={"processing_time": [1], "release_time": [5]},
    )
    env.reset()

    env.step(("noop",))

    assert env.get_action_support() == [0]


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_custom_instruction_places_through_the_transition(
    transition: str,
) -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        objective=Makespan(),
        transition=transition,
        instance={"processing_time": [3, 3]},
    )
    env.reset()

    # The conflict set is on machine 0 only, and `place` checks membership
    if transition == "gt_tetris":
        with pytest.raises(RuntimeError, match="task 0 on machine 1"):
            env.step(("execute_last", 0))

        return

    env.step(("execute_last", 0))

    assert env.state.get_assignment(0) == 1


@pytest.mark.env
@pytest.mark.parametrize("seed", range(4))
def test_whole_conflict_set_is_giffler_thompson(seed: int) -> None:
    instance = random_job_shop(Random(seed), 3, 3)
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition="gt_tetris",
        instance=instance,
    )

    assert reachable_schedules(
        _delta_gt_env(instance, 1.0)
    ) == reachable_schedules(env)


@pytest.mark.env
@pytest.mark.parametrize("seed", range(4))
def test_narrower_conflict_sets_reach_fewer_schedules(seed: int) -> None:
    instance = random_job_shop(Random(seed), 3, 3)

    narrow, half, whole = (
        reachable_schedules(_delta_gt_env(instance, delta))
        for delta in (0.0, 0.5, 1.0)
    )

    assert narrow <= half <= whole


@pytest.mark.parametrize("delta", [-0.1, 1.5])
def test_delta_outside_the_unit_interval_is_rejected(delta: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ConflictSetCandidates(delta)


@pytest.mark.env
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rule", RULES)
def test_earliest_start_candidates_dispatch_like_the_event_clock(
    rule: str, seed: int
) -> None:
    instance = random_job_shop(Random(seed), 6, 4)

    assert _dispatched_starts(instance, "nd_tetris", rule) == (
        _dispatched_starts(instance, "des", rule)
    )

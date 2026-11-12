import pickle
from collections.abc import Callable
from copy import deepcopy
from random import Random
from typing import TypeAlias

import pytest

from cpscheduler.environment import (
    FlowShopSetup,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    Makespan,
    OpenShopSetup,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
    TotalCompletionTime,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
)
from tests.schedule_checks import (
    TRANSITIONS,
    duration_errors,
    job_order_violations,
    machine_conflicts,
    overlap,
    random_job_shop,
    run_random_episode,
    schedule_of,
)

N_EPISODES = 25

ScheduleCheck: TypeAlias = Callable[[int, str], None]


def _check_job_shop(seed: int, transition: str) -> None:
    rng = Random(seed)
    instance = random_job_shop(rng, 4, 3)
    release = [rng.randint(0, 15) for _ in instance["job"]]

    env = SchedulingEnv(
        JobShopSetup(),
        [ReleaseDateConstraint()],
        Makespan(),
        transition=transition,
        instance={**instance, "release_time": release},
    )
    run_random_episode(env, rng)
    schedule = schedule_of(env.state)

    assert len(schedule) == 12
    assert not machine_conflicts(schedule)
    assert not duration_errors(
        schedule, lambda task, _: instance["processing_time"][task]
    )
    assert not job_order_violations(
        schedule, instance["job"], instance["operation"]
    )
    assert env.objective.value == max(
        scheduled.end for scheduled in schedule.values()
    )

    for task, scheduled in schedule.items():
        assert scheduled.machine == instance["machine"][task]
        assert scheduled.start >= release[task]


def _check_open_shop(seed: int, transition: str) -> None:
    rng = Random(seed)
    instance = random_job_shop(rng, 3, 3)
    del instance["operation"]

    env = SchedulingEnv(
        OpenShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance=instance,
    )
    run_random_episode(env, rng)
    schedule = schedule_of(env.state)

    assert len(schedule) == 9
    assert not machine_conflicts(schedule)

    for i in schedule:
        for j in schedule:
            if i < j and instance["job"][i] == instance["job"][j]:
                assert not overlap(schedule[i], schedule[j])


def _check_flow_shop(seed: int, transition: str) -> None:
    rng = Random(seed)
    operations = [m for _ in range(3) for m in range(3)]

    env = SchedulingEnv(
        FlowShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance={
            "job": [j for j in range(3) for _ in range(3)],
            "operation": operations,
            "processing_time": [rng.randint(1, 9) for _ in range(9)],
        },
    )
    run_random_episode(env, rng)
    schedule = schedule_of(env.state)

    assert len(schedule) == 9
    assert not machine_conflicts(schedule)
    assert all(schedule[t].machine == operations[t] for t in schedule)


def _check_identical(seed: int, transition: str) -> None:
    rng = Random(seed)
    release = [rng.randint(0, 10) for _ in range(8)]

    env = SchedulingEnv(
        IdenticalParallelMachineSetup(3),
        [ReleaseDateConstraint()],
        Makespan(),
        transition=transition,
        instance={
            "processing_time": [rng.randint(1, 9) for _ in range(8)],
            "release_time": release,
        },
    )
    run_random_episode(env, rng)
    schedule = schedule_of(env.state)

    assert len(schedule) == 8
    assert not machine_conflicts(schedule)
    assert all(schedule[t].start >= release[t] for t in schedule)


def _check_uniform(seed: int, transition: str) -> None:
    rng = Random(seed)
    processing = [rng.randint(1, 9) for _ in range(6)]
    speed = [1, 3]

    env = SchedulingEnv(
        UniformParallelMachineSetup(speed=speed),
        objective=Makespan(),
        transition=transition,
        instance={"processing_time": processing},
    )
    run_random_episode(env, rng)
    schedule = schedule_of(env.state)

    assert not machine_conflicts(schedule)
    assert not duration_errors(
        schedule, lambda task, machine: -(-processing[task] // speed[machine])
    )


def _check_unrelated(seed: int, transition: str) -> None:
    rng = Random(seed)
    processing = [[rng.randint(1, 9) for _ in range(2)] for _ in range(6)]

    env = SchedulingEnv(
        UnrelatedParallelMachineSetup(),
        objective=Makespan(),
        transition=transition,
        instance={"processing_time": processing},
    )
    run_random_episode(env, rng)
    schedule = schedule_of(env.state)

    assert not machine_conflicts(schedule)
    assert not duration_errors(
        schedule, lambda task, machine: processing[task][machine]
    )


CHECKS: dict[str, ScheduleCheck] = {
    "job_shop": _check_job_shop,
    "open_shop": _check_open_shop,
    "flow_shop": _check_flow_shop,
    "identical": _check_identical,
    "uniform": _check_uniform,
    "unrelated": _check_unrelated,
}


@pytest.mark.env
@pytest.mark.parametrize("transition", TRANSITIONS)
@pytest.mark.parametrize("setup", CHECKS)
def test_masked_actions_lead_to_valid_schedules(
    setup: str, transition: str
) -> None:
    for seed in range(N_EPISODES):
        CHECKS[setup](seed, transition)


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_transition_is_empty_before_any_reset(transition: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(), objective=Makespan(), transition=transition
    )

    assert env.transition.is_empty()


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_copies_continue_the_episode_like_the_original(transition: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=TotalCompletionTime(),
        transition=transition,
        instance={"processing_time": [1, 2]},
    )
    env.reset()
    env.step(("execute", 0))

    copies = [deepcopy(env), pickle.loads(pickle.dumps(env)), env]

    for copy in copies:
        copy.step(("execute", 1))

    assert [copy.objective.value for copy in copies] == [4, 4, 4]
    assert [copy.state for copy in copies[:2]] == [env.state, env.state]

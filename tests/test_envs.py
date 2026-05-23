from copy import deepcopy

import pytest
from common import ENV_CASES, TEST_INSTANCES, env_setup

from cpscheduler.environment import IdenticalParallelMachineSetup, SchedulingEnv
from cpscheduler.environment.constants import Status
from cpscheduler.environment.des import Schedule
from cpscheduler.environment.des.events import CheckpointEvent, SubmitEvent

ENV_CASE_KEYS: list[str] = list(ENV_CASES)


def test_empty_instance() -> None:
    env = SchedulingEnv(IdenticalParallelMachineSetup(2))

    env.load_instance({"processing_time": []})
    env.reset()
    assert env.state.is_terminal()


@pytest.mark.env
@pytest.mark.parametrize("case_name", ENV_CASE_KEYS)
def test_execute(case_name: str) -> None:
    env = ENV_CASES[case_name]()

    env.reset()

    first_action = ("execute", 0)
    obs, _, terminated, truncated, info = env.step(first_action)

    assert not terminated
    assert not truncated
    assert info["current_time"] == 0
    assert obs.task["status"][0] == Status.EXECUTING

    advancing_time = max(obs["task"]["processing_time"])
    new_obs, _, new_terminated, new_truncated, new_info = env.step(
        [("advance", advancing_time)]
    )

    assert not new_terminated
    assert not new_truncated
    assert new_info["current_time"] == advancing_time
    assert new_obs.task["status"][0] == Status.COMPLETED


@pytest.mark.env
@pytest.mark.parametrize("case_name", ENV_CASE_KEYS)
def test_reward(case_name: str) -> None:
    env = ENV_CASES[case_name]()

    env.reset()

    action = [("submit", i) for i in range(env.state.n_tasks)]
    _, reward, terminated, _, info = env.step(action)

    assert terminated
    assert info["current_time"] == -reward


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_submit(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    actions = [
        ("submit", 2),
        ("submit", 1),
        ("submit", 0),
    ]

    obs, *_, info = env.step(actions)

    assert obs.task["status"][0] == Status.COMPLETED
    assert obs.task["status"][1] == Status.COMPLETED

    assert obs.task["status"][2] == Status.EXECUTING

    assert info["current_time"] == env.state.get_end(1)

    new_obs, *_, info = env.step([("complete", 2)])

    assert new_obs.task["status"][0] == Status.COMPLETED
    assert new_obs.task["status"][1] == Status.COMPLETED
    assert new_obs.task["status"][2] == Status.COMPLETED

    assert info["current_time"] == env.state.get_end(2)


@pytest.mark.env
@pytest.mark.parametrize("case_name", ENV_CASE_KEYS)
def test_execute2(case_name: str) -> None:
    env = ENV_CASES[case_name]()

    env.reset()

    actions = [("execute", i) for i in range(env.state.n_tasks)]

    obs, _, terminated, *_ = env.step(actions)

    assert obs.task["status"] == [Status.COMPLETED] * obs.n_tasks
    assert terminated


@pytest.mark.env
@pytest.mark.parametrize("case_name", ENV_CASE_KEYS)
def test_submit2(case_name: str) -> None:
    env = ENV_CASES[case_name]()

    env.reset()

    actions = [("submit", i) for i in range(env.state.n_tasks - 1, -1, -1)]

    obs, _, terminated, *_ = env.step(actions)

    assert obs.task["status"] == [Status.COMPLETED] * obs.n_tasks
    assert terminated


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_blocking_instruction(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    # Inverse order of execution (1 requires 0 to be completed first
    deadlock_action = [("execute", 1), ("execute", 0)]

    with pytest.raises(
        RuntimeError,
        match=r"is potentially deadlocking the event queue due to an action-dependent dependency that may never happen.",
    ):
        env.step(deadlock_action)


def test_copy() -> None:
    env = env_setup("ta01")
    env.reset()

    env_copy = deepcopy(env)

    assert env is not env_copy
    # assert EzPickle.__eq__(env, env_copy)
    assert env.state == env_copy.state

    env.step(("execute", 0))
    assert env.state != env_copy.state

    env_copy.step(("execute", 0))
    assert env.state == env_copy.state


def test_pickle_roundtrip() -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(n_machines=2, disjunctive=False),
        instance={"processing_time": [2]},
    )
    env.reset()

    schedule = Schedule()
    schedule.add_event(SubmitEvent(0), env.state)
    schedule.add_event(CheckpointEvent(), env.state, time=3)

    roundtrip_schedule = deepcopy(schedule)

    assert roundtrip_schedule.next_time() == 0

    start_events = list(roundtrip_schedule.peek_events_at_time(0))
    checkpoint_events = list(roundtrip_schedule.peek_events_at_time(3))

    assert len(start_events) == 1
    assert len(checkpoint_events) == 1
    assert isinstance(start_events[0], SubmitEvent)
    assert isinstance(checkpoint_events[0], CheckpointEvent)

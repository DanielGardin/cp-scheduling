import pytest

from copy import deepcopy

from common import env_setup, TEST_INSTANCES

from cpscheduler.environment.constants import Status

@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_reset(instance_name: str) -> None:
    env = env_setup(instance_name)

    (obs, _), info = env.reset()

    assert not all(obs["available"])

    assert info["current_time"] == 0
    assert obs["status"][0] == Status.AWAITING
    assert obs["available"][0]
    assert obs["status"][1] == Status.AWAITING
    assert not obs["available"][1]

    env.step(
        [
            ("execute", 0),
            ("submit", 1),
            ("advance", 12),
            ("submit", 13),
            ("complete", 1),
        ]
    )

    (new_obs, _), new_info = env.reset()

    assert new_info["current_time"] == 0
    assert new_obs["status"][0] == Status.AWAITING
    assert new_obs["available"][0]
    assert new_obs["status"][1] == Status.AWAITING
    assert not new_obs["available"][1]


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_execute(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    first_action = ("execute", 0)
    (obs, _), _, terminated, truncated, info = env.step(first_action)

    assert not terminated
    assert not truncated
    assert info["current_time"] == 0
    assert obs["status"][0] == Status.EXECUTING

    advancing_time = max(obs["processing_time"])
    (new_obs, _), _, new_terminated, new_truncated, new_info = env.step(
        [("advance", advancing_time)]
    )

    assert not new_terminated
    assert not new_truncated
    assert new_info["current_time"] == advancing_time
    assert new_obs["status"][0] == Status.COMPLETED


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

    (obs, _), *_, info = env.step(actions)

    assert obs["status"][0] == Status.COMPLETED
    assert obs["status"][1] == Status.COMPLETED

    assert obs["status"][2] == Status.EXECUTING

    assert info["current_time"] == env.state.runtime_state.get_end(1)

    (new_obs, _), *_, info = env.step([("complete", 2)])

    assert new_obs["status"][0] == Status.COMPLETED
    assert new_obs["status"][1] == Status.COMPLETED
    assert new_obs["status"][2] == Status.COMPLETED

    assert info["current_time"] == env.state.runtime_state.get_end(2)


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_execute2(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    actions = [("execute", i) for i in range(env.state.n_tasks)]

    (obs, _), _, terminated, *_ = env.step(actions)

    assert obs["status"] == [Status.COMPLETED] * env.state.n_tasks
    assert terminated


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_submit2(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    actions = [("submit", i) for i in range(env.state.n_tasks-1, -1, -1)]

    (obs, _), _, terminated, *_ = env.step(actions)

    assert obs["status"] == [Status.COMPLETED] * env.state.n_tasks
    assert terminated

@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_blocking_instruction(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    # Test when the action cannot be done, limit the execution to 1 second and failt after that

    (obs, _), reward, *_ = env.step(
        [("execute", 1), ("execute", 0)] # Inverse order of execution (1 requires 0 to be completed first)
    )

    assert obs["status"][0] == Status.AWAITING
    assert obs["status"][1] == Status.AWAITING
    assert reward == 0

def test_copy() -> None:
    env = env_setup("ta01")

    env.reset()

    env_copy = deepcopy(env)

    assert env.state == env_copy.state

    env.step(("execute", 0))

    assert env.state != env_copy.state

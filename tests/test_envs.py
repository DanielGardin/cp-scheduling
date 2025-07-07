import pytest

from typing_extensions import Unpack

from cpscheduler.instances import generate_taillard_instance
from cpscheduler.environment import (
    SchedulingEnv,
    ResourceConstraint,
    SingleMachineSetup,
)

from common import env_setup, TEST_INSTANCES


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_reset(instance_name: str) -> None:
    env = env_setup(instance_name)

    (obs, _), info = env.reset()

    assert info["current_time"] == 0
    assert obs["status"][0] == "available"
    assert obs["status"][1] == "awaiting"

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
    assert new_obs["status"][0] == "available"
    assert new_obs["status"][1] == "awaiting"


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_execute(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    first_action = ("execute", 0)
    (obs, _), reward, terminated, truncated, info = env.step(first_action)

    assert not terminated
    assert not truncated
    assert info["current_time"] == 0
    assert obs["status"][0] == "executing"

    advancing_time = max(obs["processing_time"])
    (new_obs, _), new_reward, new_terminated, new_truncated, new_info = env.step(
        [("advance", advancing_time)]
    )

    assert not new_terminated
    assert not new_truncated
    assert new_info["current_time"] == advancing_time
    assert new_reward == -int(env.tasks[0].get_duration())
    assert new_obs["status"][0] == "completed"


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

    (obs, _), reward, terminated, truncated, info = env.step(actions)

    assert obs["status"][0] == "completed"
    assert obs["status"][1] == "completed"

    assert obs["status"][2] == "executing"

    assert (
        info["current_time"]
        == env.tasks[0].get_duration() + env.tasks[1].get_duration()
    )

    (new_obs, _), new_reward, new_terminated, new_truncated, new_info = env.step(
        [("complete", 2)]
    )

    assert new_obs["status"][0] == "completed"
    assert new_obs["status"][1] == "completed"
    assert new_obs["status"][2] == "completed"

    assert new_info["current_time"] == env.tasks[2].get_end()
    assert new_reward + reward == -int(env.tasks[2].get_end())


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_execute2(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    actions = [("execute", i) for i in range(len(env.tasks))]

    (obs, _), reward, terminated, truncated, info = env.step(actions)

    assert obs["status"] == ["completed"] * len(env.tasks)
    assert terminated


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_submit2(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    actions = [("submit", i) for i in range(len(env.tasks) - 1, -1, -1)]

    (obs, _), reward, terminated, truncated, info = env.step(actions)

    assert obs["status"] == ["completed"] * len(env.tasks)
    assert terminated


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_pause(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    processing_time = env.data["processing_time"][0]

    actions: list[tuple[str, Unpack[tuple[int, ...]]]] = [
        ("execute", 0),
        ("advance", processing_time // 2),
        ("pause", 0),
        ("advance", processing_time // 2),
        ("query",),
        ("execute", 0),
        ("complete", 0),
    ]

    (obs, _), reward, terminated, truncated, info = env.step(actions)

    assert obs["status"][0] == "paused"
    assert info["current_time"] == 2 * (processing_time // 2)
    # assert obs['remaining_time'][0] == processing_time - processing_time//2

    (new_obs, _), new_reward, new_terminated, new_truncated, new_info = env.step()

    assert new_obs["status"][0] == "completed"
    assert new_info["current_time"] == processing_time + processing_time // 2

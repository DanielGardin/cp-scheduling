from copy import deepcopy

import pytest

from cpscheduler.environment import IdenticalParallelMachineSetup, SchedulingEnv
from tests.conftest import ENV_CASES, TEST_INSTANCES, env_setup

ENV_CASE_KEYS: list[str] = list(ENV_CASES)


def test_empty_instance() -> None:
    env = SchedulingEnv(IdenticalParallelMachineSetup(2))

    env.load_instance({"processing_time": []})
    env.reset()
    assert env.state.is_terminal()


@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_blocking_instruction(instance_name: str) -> None:
    env = env_setup(instance_name)

    env.reset()

    # Inverse order of execution (1 requires 0 to be completed first
    deadlock_action = [("execute", 1), ("execute", 0)]

    with pytest.raises(
        RuntimeError,
        match=r"is potentially deadlocking the event queue",
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

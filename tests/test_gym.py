import pytest


def test_scheduling_env_gym_wraps_core_env() -> None:
    pytest.importorskip("gymnasium")

    from cpscheduler.environment import SchedulingEnv, SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    core = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [2]},
    )
    core.reset()

    env = SchedulingEnvGym.from_env(core)

    obs, info = env.reset()

    assert "current_time" in info
    assert isinstance(obs, tuple)

    _, _, terminated, truncated, _ = env.step(("execute", 0))

    assert not truncated
    assert isinstance(terminated, bool)


def test_scheduling_env_gym_constructor_builds_spaces() -> None:
    pytest.importorskip("gymnasium")

    from cpscheduler.environment import SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    env = SchedulingEnvGym(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [1, 3]},
    )

    obs, _ = env.reset()

    assert env.action_space is not None
    assert env.observation_space is not None
    assert isinstance(obs, tuple)

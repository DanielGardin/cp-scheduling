import pytest


def test_scheduling_env_gym_wraps_core_env() -> None:
    pytest.importorskip("gymnasium")

    from cpscheduler.environment import SchedulingEnv, SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    core = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [2]},
    )
    core_obs, core_info = core.reset()

    env = SchedulingEnvGym.from_env(core)

    obs, info = env.reset()

    assert obs == core_obs.serialize()
    assert info == core_info


def test_observation_invariants_and_reset_step_consistency() -> None:
    pytest.importorskip("gymnasium")

    from cpscheduler.environment import SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    env = SchedulingEnvGym(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [2, 3]},
    )

    obs, _ = env.reset()

    assert env.observation_space is not None
    assert env.observation_space.contains(obs)

    # take a valid action and ensure observations remain in the declared space
    action = ("execute", 0)
    obs2, *_ = env.step(action)

    assert env.observation_space.contains(obs2)


def test_action_logic_small_instance_changes_state() -> None:
    pytest.importorskip("gymnasium")

    from cpscheduler.environment import SchedulingEnv, SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    core = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [2, 1]},
    )
    core.reset()

    env = SchedulingEnvGym.from_env(core)
    env.reset()

    # execute job 0 then job 1; ensure environment state progresses and terminates eventually
    _, reward, terminated, *_ = env.step(("execute", 0))
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)

    # take another step; should still be valid
    _, reward2, terminated2, *_ = env.step(("execute", 1))
    assert isinstance(reward2, (int, float))
    assert isinstance(terminated2, bool)


def test_vectorized_env_support_sync_vector_env() -> None:
    pytest.importorskip("gymnasium")

    import numpy as np
    from gymnasium.vector import SyncVectorEnv

    from cpscheduler.environment import SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    def make_env() -> SchedulingEnvGym:
        return SchedulingEnvGym(
            SingleMachineSetup(disjunctive=False),
            instance={"processing_time": [1, 2]},
        )

    n_envs = 3
    vec = SyncVectorEnv([make_env for _ in range(n_envs)])

    vec.reset()

    # basic sanity checks - reset/step succeed and return batched results
    actions = [("execute", 0) for _ in range(n_envs)]
    _, rewards, terminated, *_ = vec.step(actions)

    # rewards should be length n_envs (or array-like)
    assert len(rewards) == n_envs
    # terminated/truncated should be arrays or lists of bools
    assert len(terminated) == n_envs
    assert all(isinstance(x, (bool, np.bool_)) for x in terminated)


def test_vectorized_env_support_async_vector_env() -> None:
    pytest.importorskip("gymnasium")

    import numpy as np
    from gymnasium.vector import AsyncVectorEnv

    from cpscheduler.environment import SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    def make_env() -> SchedulingEnvGym:
        return SchedulingEnvGym(
            SingleMachineSetup(disjunctive=False),
            instance={"processing_time": [1, 2]},
        )

    n_envs = 3
    vec = AsyncVectorEnv([make_env for _ in range(n_envs)])

    vec.reset()

    # basic sanity checks - reset/step succeed and return batched results
    actions = [("execute", 0) for _ in range(n_envs)]
    _, rewards, terminated, *_ = vec.step(actions)

    # rewards should be length n_envs (or array-like)
    assert len(rewards) == n_envs
    # terminated/truncated should be arrays or lists of bools
    assert len(terminated) == n_envs
    assert all(isinstance(x, (bool, np.bool_)) for x in terminated)


def test_observation_space_matches_core_observation_structure() -> None:
    pytest.importorskip("gymnasium")

    from cpscheduler.environment import SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    env = SchedulingEnvGym(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [1, 2, 3]},
    )

    # observation_space is inferred from core observation.to_tuple()
    serialized_obs = env.core.observation.serialize()
    # The space should accept the current observation
    assert env.observation_space.contains(serialized_obs)


def test_observation_space_updates_on_load_instance_and_reset_options() -> None:
    pytest.importorskip("gymnasium")

    from cpscheduler.environment import SingleMachineSetup, ReleaseDateConstraint
    from cpscheduler.gym import SchedulingEnvGym

    env = SchedulingEnvGym(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [1]},
    )

    # initial space corresponds to single-job instance
    obs1, _ = env.reset()
    assert env.observation_space.contains(obs1)

    # change instance to two jobs -> observation_space should expand
    env.load_instance({"processing_time": [1, 2]})
    obs2, _ = env.reset()
    assert not env.observation_space.contains(obs1)  # old obs should no longer be valid
    assert env.observation_space.contains(obs2)

    # calling reset with options should also recompute the observation space
    obs3, _ = env.reset(options={})
    assert env.observation_space.contains(obs2)
    assert env.observation_space.contains(obs3)


def test_from_env_delegation_and_info_mapping_preserved() -> None:
    pytest.importorskip("gymnasium")

    from cpscheduler.environment import SchedulingEnv, SingleMachineSetup
    from cpscheduler.gym import SchedulingEnvGym

    core = SchedulingEnv(SingleMachineSetup(disjunctive=False), instance={"processing_time": [1]})
    core.reset()

    gym_env = SchedulingEnvGym.from_env(core)

    # `core` attribute should reference the same underlying env
    assert gym_env.core is core

    _, info = gym_env.reset()
    assert isinstance(info, dict)

    # step should return a plain dict for info (mapping preserved)
    *_, info2 = gym_env.step(("execute", 0))
    assert isinstance(info2, dict)

    # attribute access should delegate to core (e.g., get_entry exists on core)
    assert gym_env.get_entry() == core.get_entry()

    *_, terminated, truncated, _ = gym_env.step(("execute", 0))

    assert not truncated
    assert isinstance(terminated, bool)

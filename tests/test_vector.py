from typing import Any, Callable
from pandas import DataFrame

import pytest

import numpy as np

from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from cpscheduler.environment import SchedulingCPEnv
from cpscheduler.instances.jobshop import generate_taillard_instance
from cpscheduler.policies.heuristics import ShortestProcessingTime, MostOperationsRemaining, MostWorkRemaining, PriorityDispatchingRule

from common import env_setup

@pytest.mark.vector
def test_sync_env() -> None:

    def make_env_fn(instance_name: str) -> Callable[[], SchedulingCPEnv]:
        def env_fn() -> SchedulingCPEnv:
            return env_setup(instance_name)

        return env_fn

    # Observation space is the same

    sync_env = SyncVectorEnv([
        make_env_fn(instance_name) for instance_name in [f"ta{i:02d}" for i in range(1, 11)]
    ])

    envs = [
        make_env_fn(instance_name)() for instance_name in [f"ta{i:02d}" for i in range(1, 11)]
    ]

    sync_obs, sync_info = sync_env.reset() # type: ignore

    for i, env in enumerate(envs):
        single_obs, single_info = env.reset()
    
        assert all([np.all(single_obs[feat] == sync_obs[feat][i]) for feat in single_obs])

    pdr = ShortestProcessingTime()

    actions = [
        pdr({feat: sync_obs[feat][i] for feat in sync_obs}) for i in range(len(envs))
    ]

    final_obss, final_rewards, final_terminated, final_truncated, final_infos = sync_env.step(actions) # type: ignore

    for i in range(len(envs)):
        single_new_obs, single_reward, single_terminated, single_truncated, single_new_info = envs[i].step(actions[i])

        assert all([np.all(single_new_obs[feat] == final_obss[feat][i]) for feat in single_new_obs])
        assert final_rewards[i] == single_reward
        assert final_terminated[i] == single_terminated
        assert final_truncated[i] == single_truncated


@pytest.mark.vector
def test_async_env() -> None:

    def make_env_fn(instance_name: str) -> Callable[[], SchedulingCPEnv]:
        def env_fn() -> SchedulingCPEnv:
            return env_setup(instance_name)
        
        return env_fn

    # Observation space is the same

    sync_env = AsyncVectorEnv([
        make_env_fn(instance_name) for instance_name in [f"ta{i:02d}" for i in range(1, 11)]
    ], shared_memory=False) # TODO: verify why shared_memory=True is not working

    envs = [
        make_env_fn(instance_name)() for instance_name in [f"ta{i:02d}" for i in range(1, 11)]
    ]

    sync_obs, sync_info = sync_env.reset() # type: ignore

    for i, env in enumerate(envs):
        single_obs, single_info = env.reset()
    
        assert all([np.all(single_obs[feat] == sync_obs[feat][i]) for feat in single_obs])

    pdr = ShortestProcessingTime()

    actions = [
        pdr({feat: sync_obs[feat][i] for feat in sync_obs}) for i in range(len(envs))
    ]

    final_obss, final_rewards, final_terminated, final_truncated, final_infos = sync_env.step(actions) # type: ignore

    for i in range(len(envs)):
        single_new_obs, single_reward, single_terminated, single_truncated, single_new_info = envs[i].step(actions[i])

        assert all([np.all(single_new_obs[feat] == final_obss[feat][i]) for feat in single_new_obs])
        assert final_rewards[i] == single_reward
        assert final_terminated[i] == single_terminated
        assert final_truncated[i] == single_truncated
from typing import Any, Callable
from pandas import DataFrame

import pytest

import numpy as np

from cpscheduler.common_envs import JobShopEnv
from cpscheduler.environment.instances import generate_taillard_instance

from cpscheduler.policies.heuristics import ShortestProcessingTime, MostOperationsRemaining, MostWorkRemaining
from cpscheduler.environment.vector import SyncVectorEnv, AsyncVectorEnv, RayVectorEnv, VectorEnv

def make_jsp_env(
        instance_generator: Callable[..., tuple[DataFrame, dict[str, Any]]],
        env_cls: Callable[[DataFrame], JobShopEnv],
        *args: Any, **kwargs: Any
    ) -> Callable[[], JobShopEnv]:
    return lambda : env_cls(instance_generator(*args, **kwargs)[0])


pdrs = [
    ShortestProcessingTime(),
    MostOperationsRemaining(),
    MostWorkRemaining()
]

@pytest.mark.vector
@pytest.mark.parametrize('env_cls', [AsyncVectorEnv, SyncVectorEnv, RayVectorEnv])
def test_sync_env(env_cls: type[VectorEnv]) -> None:
    env_fns = [
        make_jsp_env(
            generate_taillard_instance,
            JobShopEnv,
            15, 15, seed=i
        ) for i in range(3)
    ]

    sync_env = env_cls(env_fns, auto_reset=False)
    envs     = [env_fn() for env_fn in env_fns]

    obss, infos = sync_env.reset()

    for env, obs, info in zip(envs, obss, infos):
        single_obs, single_info = env.reset()

        assert np.all(obs == single_obs)
        # Fix me: checking equality of empty np.array results in False
        # assert info == single_info
    

    actions = [pdr(obs) for pdr, obs in zip(pdrs, obss)]

    final_obss, final_rewards, final_terminated, final_truncated, final_infos = sync_env.step(actions, enforce_order=False)

    for i in range(len(envs)):
        single_new_obs, single_reward, single_terminated, single_truncated, single_new_info = envs[i].step(actions[i], enforce_order=False)

        assert np.all(final_obss[i] == single_new_obs)
        assert final_rewards[i] == single_reward
        assert final_terminated[i] == single_terminated
        assert final_truncated[i] == single_truncated
        # assert final_infos[i] == single_new_info
    
    sync_env = SyncVectorEnv(env_fns, auto_reset=True)

    obss, _ = sync_env.reset()

    actions = [pdr(obs) for pdr, obs in zip(pdrs, obss)]

    new_obss, rewards, terminated, truncated, new_infos = sync_env.step(actions, enforce_order=False)

    assert all([np.all(new_obs == obs) for obs, new_obs in zip(obss, new_obss)])
    assert 'final_obs' in new_infos
    assert 'final_reward' in new_infos
    assert 'final_terminated' in new_infos
    assert 'final_truncated' in new_infos
    assert 'final_info' in new_infos

    assert new_infos['final_info'].keys() == final_infos.keys()

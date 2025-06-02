from typing import Any, Sequence

from ..env import Env, _Obs, _Action

def step_with_autoreset(
        env: Env[_Obs, _Action],
        action: _Action,
        *args: Any, **kwargs: Any
    ) -> tuple[_Obs, float, bool, bool, dict[str, Any]]:
    obs, reward, terminated, truncated, info = env.step(action, *args, **kwargs)

    if terminated:
        old_info = info
        obs, info = env.reset()

        info.update({
            'final_obs'        : obs,
            'final_reward'     : reward,
            'final_terminated' : terminated,
            'final_truncated'  : truncated,
            'final_info'       : old_info,
        })

        reward = 0.
        terminated = False
        truncated = False

    return obs, reward, terminated, truncated, info


def get_attribute(env: Env[_Obs, _Action], name: str, *args: Any, **kwargs: Any) -> Any:
    attr = getattr(env, name)

    if not callable(attr):
        return attr
    
    return attr(*args, **kwargs)


def info_union(infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    known_keys: set[str] = set()
    keys: list[str] = []

    for info in infos:
        new_keys = [key for key in info.keys() if key not in known_keys]

        keys.extend(new_keys)
        known_keys.update(new_keys)

    new_info: dict[str, Any] = {key: [info.get(key, None) for info in infos] for key in keys}

    if 'final_info' in new_info:
        new_info['final_info'] = info_union([
            info if info is not None else {} for info in new_info['final_info']
        ])

    return new_info
from typing import Any, Sequence, SupportsFloat, TypeVar, Protocol, Iterable, Callable, runtime_checkable

_Obs = TypeVar('_Obs', covariant=True)
_Act = TypeVar('_Act', contravariant=True)
@runtime_checkable
class Env(Protocol[_Obs, _Act]):
    def reset(self) -> tuple[_Obs, dict[str, Any]]: ...

    def step(self, action: _Act, *args: Any, **kwargs: Any) -> tuple[_Obs, SupportsFloat, bool, bool, dict[str, Any]]: ...

    def render(self) -> Any: ...

def step_with_autoreset(
        env: Env[_Obs, _Act],
        action: _Act,
        *args: Any, **kwargs: Any
    ) -> tuple[_Obs, SupportsFloat, bool, bool, dict[str, Any]]:
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


_SingleObs = TypeVar('_SingleObs')
_SingleAct = TypeVar('_SingleAct', contravariant=True)
class VectorEnv(Protocol[_SingleObs, _SingleAct]):
    def __init__(
            self,
            env_fns: Iterable[Callable[[], Env[_SingleObs, _SingleAct]]],
            *args: Any, **kwargs: Any
        ) -> None: ...

    def reset(self) -> tuple[list[_SingleObs], dict[str, Any]]: ...

    def step(self, actions: Iterable[_SingleAct], *args: Any, **kwargs: Any) -> tuple[list[_SingleObs], list[SupportsFloat], list[bool], list[bool], dict[str, Any]]: ...

    def render(self) -> None: ...

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]: ...

    def close(self) -> None:
        return


def get_attribute(env: Env[_Obs, _Act], name: str, *args: Any, **kwargs: Any) -> Any:
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
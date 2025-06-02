from typing import Any, Callable, Iterable, TypeVar, SupportsFloat

from copy import deepcopy

from .common import Env, VectorEnv, step_with_autoreset, get_attribute, info_union


_Obs = TypeVar('_Obs')
_Act = TypeVar('_Act')
class SyncVectorEnv(VectorEnv[_Obs, _Act]):
    def __init__(
            self,
            env_fns: Iterable[Callable[[], Env[_Obs, _Act]]],
            copy: bool = False,
            auto_reset: bool = True,
        ):
        self.env_fns    = env_fns
        self.copy       = copy
        self.auto_reset = auto_reset


        self.envs = [env_fn() for env_fn in env_fns]
        self.n_envs = len(self.envs)


    def reset(self) -> tuple[list[_Obs], dict[str, Any]]:
        obs, infos = map(list, zip(*[env.reset() for env in self.envs]))

        if self.copy:
            obs = deepcopy(obs)

        return obs, info_union(infos)


    def step(self, actions: Iterable[_Act], *args: Any, **kwargs: Any) -> tuple[list[_Obs], list[SupportsFloat], list[bool], list[bool], dict[str, Any]]:
        if self.auto_reset:
            obs, rewards, terminated, truncated, infos = map(
                list, zip(*[step_with_autoreset(env, action, *args, **kwargs) for env, action in zip(self.envs, actions)])
            )

        else:
            obs, rewards, terminated, truncated, infos = map(
                list, zip(*[env.step(action, *args, **kwargs) for env, action in zip(self.envs, actions)])
            )

        return obs, rewards, terminated, truncated, info_union(infos)


    def render(self) -> None:
        for env in self.envs:
            env.render()


    def call(self, name: str, *args :Any, **kwargs: Any) -> tuple[list[Any], ...]:
        results = [get_attribute(env, name, *args, **kwargs) for env in self.envs]

        return tuple(map(list, zip(*results)))

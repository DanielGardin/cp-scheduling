from typing import Any, Callable, Iterable

from copy import deepcopy

from ..env import Env
from .common import step_with_autoreset, get_attribute, info_union

class SyncVectorEnv:
    def __init__(
            self,
            env_fns: Iterable[Callable[[], Env]],
            copy: bool = False,
            auto_reset: bool = True,
        ):
        self.env_fns    = env_fns
        self.copy       = copy
        self.auto_reset = auto_reset


        self.envs = [env_fn() for env_fn in env_fns]
        self.n_envs = len(self.envs)


    def reset(self) -> tuple[list[Any], dict[str, Any]]:
        obs, infos = map(list, zip(*[env.reset() for env in self.envs]))

        if self.copy:
            obs = deepcopy(obs)

        return obs, info_union(infos)


    def step(self, actions: Iterable[Any], *args: Any, **kwargs: Any) -> tuple[list[Any], list[float], list[bool], list[bool], dict[str, Any]]:
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


    def call(self, name: str, *args :Any, **kwargs: Any) -> tuple[Any, ...]:
        return tuple([get_attribute(env, name, *args, **kwargs) for env in self.envs])
from typing import Any, TypeVar, Iterable, Callable, Sequence, SupportsFloat

import ray
from copy import deepcopy

from .common import Env, VectorEnv, step_with_autoreset, get_attribute, info_union


_Obs = TypeVar('_Obs')
_Act = TypeVar('_Act')
@ray.remote
class RayEnvWorker(Env[_Obs, _Act]):
    def __init__(self, env_fn: Callable[[], Env[_Obs, _Act]], auto_reset: bool):
        self.env = env_fn()
        self.auto_reset = auto_reset


    def reset(self) -> tuple[_Obs, dict[str, Any]]:
        return self.env.reset()


    def step(self, action: _Act, *args: Any, **kwargs: Any) -> tuple[_Obs, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.auto_reset:
            return step_with_autoreset(self.env, action, *args, **kwargs)

        return self.env.step(action, *args, **kwargs)


    def render(self) -> None:
        self.env.render()


    def close(self) -> None:
        del self.env


    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ["reset", "step", "close", "render"]:
            raise ValueError(f'Trying to call function `{name}` with `call`, use `{name}` directly instead.')
        return get_attribute(self.env, name, *args, **kwargs)


class RayVectorEnv(VectorEnv[_Obs, _Act]):
    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env[_Obs, _Act]]],
        copy: bool = False,
        auto_reset: bool = True,
    ):
        self.env_fns = env_fns
        self.copy = copy
        self.auto_reset = auto_reset

        self.workers = [
            RayEnvWorker.remote(env_fn, auto_reset) for env_fn in env_fns # type: ignore
        ]

        self.n_envs = len(self.workers)


    def handle_errors(self, results: Any, successes: list[bool]) -> None:
        if not all(successes):
            raise ValueError(f'Error in operation: {results}')


    def reset(self) -> tuple[list[_Obs], dict[str, Any]]:
        results = ray.get([worker.reset.remote() for worker in self.workers]) # type: ignore
        obs, infos = map(list, zip(*results))

        if self.copy:
            obs = deepcopy(obs)

        return obs, info_union(infos)


    def step(
        self, actions: Iterable[_Act], *args: Any, **kwargs: Any
    ) -> tuple[list[_Obs], list[SupportsFloat], list[bool], list[bool], dict[str, Any]]:
        if (isinstance(actions, Sequence) and len(actions) != self.n_envs) or sum(1 for _ in actions) != self.n_envs:
            raise ValueError(f'Number of actions does not match number of environments ({self.n_envs})')

        results = ray.get([
            worker.step.remote(action, *args, **kwargs) # type: ignore
            for worker, action in zip(self.workers, actions)
        ])
        obs, rewards, terminated, truncated, infos = map(list, zip(*results))

        if self.copy:
            obs = deepcopy(obs)

        return obs, rewards, terminated, truncated, info_union(infos)

    def render(self) -> None:
        ray.get([worker.render.remote() for worker in self.workers]) # type: ignore

    def close(self) -> None:
        ray.get([worker.close.remote() for worker in self.workers]) # type: ignore
        self.workers = []

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[list[Any], ...]:
        results = ray.get([
            worker.call.remote(name, *args, **kwargs) for worker in self.workers # type: ignore
        ])
        return tuple(map(list, zip(*results)))
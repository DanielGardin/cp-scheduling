from typing import Any, Protocol, Iterable, Callable

from .async_env import AsyncVectorEnv
from .sync_env import SyncVectorEnv
from .ray_env import RayVectorEnv

from ..env import Env


class VectorEnv(Protocol):
    def __init__(
            self,
            env_fns: Iterable[Callable[[], Env]],
            *args: Any, **kwargs: Any
        ) -> None: ...

    def reset(self) -> tuple[Any, dict[str, Any]]: ...

    def step(self, actions: Iterable[Any], *args: Any, **kwargs: Any) -> tuple[list[Any], list[float], list[bool], list[bool], dict[str, Any]]: ...

    def render(self) -> None: ...

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]: ...

__all__ = [
    'VectorEnv',
    'AsyncVectorEnv',
    'SyncVectorEnv',
    'RayVectorEnv'
]
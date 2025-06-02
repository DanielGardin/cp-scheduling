from typing import Any, TypeVar, ParamSpec, SupportsFloat, Protocol
from abc import ABC, abstractmethod

from .. import Env

_WrappedObs = TypeVar('_WrappedObs')
_WrappedAct = TypeVar('_WrappedAct')
_P = ParamSpec('_P')
class WrappedEnv(Env[_WrappedObs, _WrappedAct], ABC):
    def __init__(
            self,
            env: Env,
            *p_args: Any, **p_kwargs: Any
        ):
        self.env = env

    @abstractmethod
    def reset(self) -> tuple[_WrappedObs, dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: _WrappedAct, *args: Any, **kwargs: Any) -> tuple[_WrappedObs, SupportsFloat, bool, bool, dict[str, Any]]:
        ...

    def render(self) -> None:
        self.env.render()

from typing import Any, TypeVar, ParamSpec, SupportsFloat, Protocol

from .. import Env

_Obs = TypeVar('_Obs')
_Act = TypeVar('_Act')
_WrappedObs = TypeVar('_WrappedObs')
_WrappedAct = TypeVar('_WrappedAct')
_P = ParamSpec('_P')
class WrappedEnv(Env[_WrappedObs, _WrappedAct]):
    def __init__(
            self,
            env: Env[_Obs, _Act],
            *p_args: Any, **p_kwargs: Any
        ):
        self.env = env

    def reset(self) -> tuple[_WrappedObs, dict[str, Any]]:
        raise NotImplementedError()

    def step(self, action: _WrappedAct, *args: Any, **kwargs: Any) -> tuple[_WrappedObs, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError()

    def render(self) -> None:
        self.env.render()

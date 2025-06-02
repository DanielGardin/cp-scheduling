from typing import Any, TypeVar, Callable, Concatenate, ParamSpec
from types import MethodType

from ..env import Env

_Env = TypeVar('_Env', bound=Env)
_Obs = TypeVar('_Obs')
_Act = TypeVar('_Act')

_P = ParamSpec('_P')
def wrap_obs(
        env: _Env,
        post_processor: Callable[Concatenate[Any, _P], _Obs],
        *args: Any,
        **kwargs: Any
    ) -> _Env:
    original_step = env.step
    original_reset = env.reset
    def new_step(self: _Env, action: Any) -> tuple[_Obs, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = original_step(action)
        return post_processor(obs, *args, **kwargs), reward, terminated, truncated, info

    def new_reset(self: _Env) -> tuple[_Obs, dict[str, Any]]:
        obs, info = original_reset()
        return post_processor(obs, *args, **kwargs), info

    setattr(env, 'step', MethodType(new_step, env))
    setattr(env, 'reset', MethodType(new_reset, env))

    return env


def wrap_action(
        env: _Env,
        pre_processor: Callable[Concatenate[_Act, _P], Any],
        *args: Any,
        **kwargs: Any
    ) -> _Env:
    original_step = env.step

    def new_step(self: _Env, action: _Act) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        new_action = pre_processor(action, *args, **kwargs)
        return original_step(new_action)

    setattr(env, 'step', MethodType(new_step, env))

    return env
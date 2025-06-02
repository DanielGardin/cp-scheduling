from typing import Any, Sequence, SupportsFloat, TypeVar, Protocol, Iterable, Callable, runtime_checkable

_Obs = TypeVar('_Obs', covariant=True)
_Act = TypeVar('_Act', contravariant=True)
@runtime_checkable
class Env(Protocol[_Obs, _Act]):
    def reset(self) -> tuple[_Obs, dict[str, Any]]: ...

    def step(self, action: _Act, *args: Any, **kwargs: Any) -> tuple[_Obs, SupportsFloat, bool, bool, dict[str, Any]]: ...

    def render(self) -> Any: ...


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

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[list[Any], ...]: ...

    def close(self) -> None:
        return


_WrappedObs = TypeVar('_WrappedObs', covariant=True)
_WrappedAct = TypeVar('_WrappedAct', contravariant=True)
_Env = TypeVar('_Env', bound=Env[Any, Any])
class WrappedEnv(Protocol[_WrappedObs, _WrappedAct, _Env]):
    env: _Env

    def __init__(
            self,
            env: _Env,
            *p_args: Any, **p_kwargs: Any
        ) -> None:
        self.env = env

    def unwrap(self) -> _Env:
        return self.env

    def reset(self) -> tuple[_WrappedObs, dict[str, Any]]: ...

    def step(self, action: _WrappedAct, *args: Any, **kwargs: Any) -> tuple[_WrappedObs, SupportsFloat, bool, bool, dict[str, Any]]: ...

    def render(self) -> None:
        self.env.render()

__all__ = [
    'Env',
    "IntervalVars",
    "SchedulingCPEnv",
    "AsyncVectorEnv",
    "RayVectorEnv",
    "read_jsp_instance",
    "PrecedenceConstraint",
    "NonOverlapConstraint",
    "ReleaseTimesConstraint",
    "DueDatesConstraint",
    "Makespan",
    "WeightedCompletionTime",
    "VectorEnv",
]

from typing import Any, Optional, overload, Sequence, Literal, Callable
from numpy.typing import NDArray
from pandas import DataFrame

import numpy as np

from .constraints import Constraint, PrecedenceConstraint, NonOverlapConstraint, ReleaseTimesConstraint, DueDatesConstraint

from .objectives import Objective, Makespan, WeightedCompletionTime

from .variables import IntervalVars

from .env import SchedulingCPEnv

from .vector import SyncVectorEnv, AsyncVectorEnv, RayVectorEnv

from .instances import read_jsp_instance

from .protocols import Env, VectorEnv, WrappedEnv

known_envs: dict[str, type[SchedulingCPEnv]] = {}

def register_env(env: type[SchedulingCPEnv], name: Optional[str] = None) -> None:
    if name is None:
        name = env.__name__

    known_envs[name] = env


# TODO: Change all NDArrays in the parameters to ArrayLike (not trivial, ArrayLike can be Sequence)

@overload
def make_env(
        name: str,
        instances: DataFrame | Sequence[DataFrame],
        durations: str | NDArray[np.int32] | Sequence[str | NDArray[np.int32]],
        num_envs: int,
        vector_env: Literal['sync'] = 'sync',
        auto_reset: bool = True,
        *args: Any, **kwargs: Any,
    ) -> SyncVectorEnv[Any, Any]: ...

@overload
def make_env(
        name: str,
        instances: DataFrame | Sequence[DataFrame],
        durations: str | NDArray[np.int32] | Sequence[str | NDArray[np.int32]],
        num_envs: int,
        vector_env: Literal['async'],
        auto_reset: bool = True,
        *args: Any, **kwargs: Any,
    ) -> AsyncVectorEnv[Any, Any]: ...

@overload
def make_env(
        name: str,
        instances: DataFrame | Sequence[DataFrame],
        durations: str | NDArray[np.int32] | Sequence[str | NDArray[np.int32]],
        num_envs: int,
        vector_env: Literal['ray'],
        auto_reset: bool = True,
        *args: Any, **kwargs: Any,
    ) -> RayVectorEnv[Any, Any]: ...


@overload
def make_env(
        name: str,
        instances: DataFrame,
        durations: str | NDArray[np.int32],
        num_envs: None = None,
        vector_env: Literal['sync', 'async', 'ray'] = 'sync',
        auto_reset: bool = True,
        *args: Any, **kwargs: Any,
    ) -> SchedulingCPEnv: ...

def make_env(
        name: str,
        instances: DataFrame | Sequence[DataFrame],
        durations: str | NDArray[np.int32] | Sequence[str | NDArray[np.int32]],
        num_envs: Optional[int] = None,
        vector_env: Literal['sync', 'async', 'ray'] = 'sync',
        auto_reset: bool = True,
        *args: Any, **kwargs: Any,
    ) -> SchedulingCPEnv | VectorEnv[Any, Any]:
    if name not in known_envs:
        raise ValueError(f'Environment {name} not registered.')
    
    if num_envs is None:
        if not isinstance(instances, DataFrame) or (isinstance(durations, Sequence) and not isinstance(durations, str)):
            raise ValueError('When `num_envs` is not provided, `instances` must be a DataFrame.')

        return known_envs[name](instances, durations, *args, **kwargs)

    env_fns = [
        lambda idx=i: known_envs[name](
            instances[idx] if isinstance(instances, Sequence) else instances,
            durations[idx] if isinstance(durations, Sequence) and not isinstance(durations, str) else durations,
            *args, **kwargs
        ) for i in range(num_envs)
    ]

    if vector_env == 'sync':
        return SyncVectorEnv(env_fns, auto_reset=auto_reset)

    if vector_env == 'async':
        return AsyncVectorEnv(env_fns, auto_reset=auto_reset)

    if vector_env == 'ray':
        return RayVectorEnv(env_fns, auto_reset=auto_reset)

    raise ValueError(f'Unknown vector environment {vector_env}. Available options are `sync`, `async` and `ray`.')

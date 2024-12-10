from typing import Any, Optional, overload, Sequence, Literal, Callable
from numpy.typing import NDArray
from pandas import DataFrame

import numpy as np

from .constraints import Constraint, PrecedenceConstraint, NonOverlapConstraint, ReleaseTimesConstraint, \
    DueDatesConstraint

from .objectives import Objective, Makespan, WeightedCompletionTime

from .variables import IntervalVars

from .env import SchedulingCPEnv, Env

from .vector import SyncVectorEnv, AsyncVectorEnv, RayVectorEnv, VectorEnv

from .instances import read_jsp_instance


__all__ = [
    "IntervalVars",
    "SchedulingCPEnv",
    "read_jsp_instance",
    "PrecedenceConstraint",
    "NonOverlapConstraint",
    "ReleaseTimesConstraint",
    "DueDatesConstraint",
    "Makespan",
    "WeightedCompletionTime",
    "Env"
]

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
    ) -> SyncVectorEnv: ...

@overload
def make_env(
        name: str,
        instances: DataFrame | Sequence[DataFrame],
        durations: str | NDArray[np.int32] | Sequence[str | NDArray[np.int32]],
        num_envs: int,
        vector_env: Literal['async'],
        auto_reset: bool = True,
        *args: Any, **kwargs: Any,
    ) -> AsyncVectorEnv: ...

@overload
def make_env(
        name: str,
        instances: DataFrame | Sequence[DataFrame],
        durations: str | NDArray[np.int32] | Sequence[str | NDArray[np.int32]],
        num_envs: int,
        vector_env: Literal['ray'],
        auto_reset: bool = True,
        *args: Any, **kwargs: Any,
    ) -> RayVectorEnv: ...


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
    ) -> SchedulingCPEnv | VectorEnv:
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


def build_env(
        instance: DataFrame,
        duration: str | NDArray[np.int32],
        objective: Objective | Callable[[IntervalVars], Objective],
        constraints: list[Constraint] | list[Callable[[IntervalVars], Constraint]] = [],
    ) -> SchedulingCPEnv:
    env = SchedulingCPEnv(instance, duration)

    if isinstance(objective, Objective):
        env.set_objective(objective)

    else:
        env.set_objective(objective(env.tasks))
    
    for constraint in constraints:
        if isinstance(constraint, Constraint):
            env.add_constraint(constraint)
        
        else:
            env.add_constraint(constraint(env.tasks))
    
    return env
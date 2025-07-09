from typing import overload, Any
from collections.abc import Mapping

import numpy as np
from gymnasium.spaces import Space, Dict, Tuple, Box, MultiBinary, Text, Sequence

from cpscheduler.environment._common import MAX_INT, MIN_INT
from cpscheduler.environment.utils import is_iterable_type

@overload
def infer_collection_space(obs: Mapping[Any, Any]) -> Dict: ...


@overload
def infer_collection_space(obs: tuple[Any, ...]) -> Tuple: ...


@overload
def infer_collection_space(obs: list[str]) -> Tuple: ...


@overload
def infer_collection_space(obs: list[int] | list[float]) -> Box: ...


@overload
def infer_collection_space(obs: list[bool]) -> MultiBinary: ...


def infer_collection_space(obs: Any) -> Space[Any]:
    "Infer the Gymnasium space of a collection based on its elements."

    if isinstance(obs, Mapping):
        return Dict(
            {key: infer_collection_space(value) for key, value in obs.items()}
        )

    if isinstance(obs, tuple):
        return Tuple([infer_collection_space(elem) for elem in obs])

    n = sum(1 for _ in obs)

    if is_iterable_type(obs, str):
        return Sequence(Text(max_length=100), stack=True)

    if is_iterable_type(obs, int):
        return Box(
            low=int(MIN_INT), high=int(MAX_INT), shape=(n,), dtype=np.int64
        )

    if is_iterable_type(obs, float):
        return Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64)

    if is_iterable_type(obs, bool):
        return MultiBinary(n)

    raise TypeError(
        f"Unsupported type {type(obs)} for inferring Gymnasium space. Is it a collection?"
    )

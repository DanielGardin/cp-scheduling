from typing import Any
from collections.abc import Mapping, Iterable, Sequence
from typing_extensions import TypeIs

from functools import singledispatch

import numpy as np

from gymnasium.spaces import Space, Dict, Tuple, Box, MultiBinary, Text

from cpscheduler.environment._common import MAX_INT, MIN_INT


def is_original_space(space: Any) -> TypeIs[Iterable[Dict]]:
    "Check if the space is the original space type (Dict of Dicts)."
    return isinstance(space, Tuple) and all(isinstance(s, Dict) for s in space.spaces)


@singledispatch
def create_list_space(elem: Any, size: int) -> Space[Any]:
    "Create a Gymnasium space for a list of a specific type and size."
    raise NotImplementedError(f"Unsupported type {type(elem)} for creating list space.")


@create_list_space.register
def _(elem: str, size: int) -> Space[tuple[str, ...]]:
    "Create a Gymnasium space for a list of strings."
    return Tuple(Text(max_length=100) for _ in range(size))


@create_list_space.register
def _(elem: int, size: int) -> Space[Iterable[int]]:
    "Create a Gymnasium space for a list of integers."
    return Box(low=int(MIN_INT), high=int(MAX_INT), shape=(size,), dtype=np.int64)


@create_list_space.register
def _(elem: float, size: int) -> Space[Iterable[float]]:
    "Create a Gymnasium space for a list of floats."
    return Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float64)


@create_list_space.register
def _(elem: bool, size: int) -> Space[Iterable[bool]]:
    "Create a Gymnasium space for a list of booleans."
    return MultiBinary(size)


# This function can be extended to support more types as needed.


def infer_collection_space(
    obs: Mapping[Any, Any] | tuple[Any, ...] | Sequence[Any],
) -> Space[Any]:
    "Infer the Gymnasium space of a collection based on its elements."

    if isinstance(obs, Mapping):
        return Dict({key: infer_collection_space(value) for key, value in obs.items()})

    if isinstance(obs, tuple):
        return Tuple(infer_collection_space(elem) for elem in obs)

    n = sum(1 for _ in obs)
    if n == 0:
        return Tuple([])

    return create_list_space(next(iter(obs)), n)

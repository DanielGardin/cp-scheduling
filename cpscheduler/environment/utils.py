"""
    utils.py

    This module provides utility functions for the environment and other modules.
"""
from typing import (
    Any,
    TypeVar,
    Optional,
    overload,
    Iterable,
    Sequence,
    Mapping
)
from typing_extensions import TypeIs

from collections import deque
from fractions import Fraction
from math import lcm

import numpy as np
from gymnasium import spaces

from .common import MAX_INT, MIN_INT

_S = TypeVar("_S")
_T = TypeVar("_T", bound=Any)
@overload
def convert_to_list(array: Any, dtype: type[_T]) -> list[_T]: ...

@overload
def convert_to_list(array: Iterable[_S], dtype: None = None) -> list[_S]: ...

@overload
def convert_to_list(array: Any, dtype: None = None) -> list[Any]: ...

def convert_to_list(
    array: Iterable[Any], dtype: Optional[type[_T]] = None
) -> list[Any]:
    """
    Convert an iterable to a list. If a dtype is provided, the elements of the list will be casted
    to that type.

    Parameters
    ----------
    array: Iterable
        The iterable to be converted to a list.

    dtype: type, optional
        The type to which the elements of the list will be casted to.

    Returns
    -------
    list
        A list containing the elements of the iterable.
    """

    if hasattr(array, "tolist"):
        array = getattr(array, "tolist")()

    try:
        if dtype is None:
            return array if isinstance(array, list) else list(array)

        return [dtype(item) for item in array]

    # If the iterable is not a collection, it will raise a TypeError
    except TypeError:
        return [array] if dtype is None else [dtype(array)]

def is_iterable_type(obj: Any, dtype: type[_T]) -> TypeIs[Iterable[_T]]:
    """
    Returns whether the object is an iterable containing elements of the specified type.

    Parameters
    ----------
    obj: Any
        The object to be checked.

    dtype: type
        The type of the elements of the iterable.

    Returns
    -------
    bool
        Whether the object is an iterable containing elements of the specified type.
    """
    try:
        return all([isinstance(item, dtype) for item in obj])

    except Exception:
        return False


_K = TypeVar("_K")
_V = TypeVar("_V")
def is_mapping(
    obj: dict[Any, Any] | Any, keys_type: type[_K], values_type: type[_V]
) -> TypeIs[Mapping[_K, _V]]:
    """
    Returns whether the object is a dictionary.

    Parameters
    ----------
    obj: Any
        The object to be checked.

    Returns
    -------
    bool
        Whether the object is a dictionary.
    """
    try:
        return all([
                isinstance(key, keys_type) and isinstance(value, values_type)
                for key, value in obj.items()
        ])

    except Exception:
        return False


def topological_sort(precedence_map: dict[int, list[int]], n_tasks: int) -> list[int]:
    """
    Perform a topological sort on a directed acyclic graph.

    Parameters
    ----------
    precedence_map: dict
        A dictionary containing the precedence relationships between the tasks.

    in_degree: list
        A list containing the in-degree of each task.

    Returns
    -------
    list
        A list containing the tasks in topological order
    """
    in_degree = [0] * n_tasks
    for children in precedence_map.values():
        for child in children:
            in_degree[child] += 1

    queue = deque([task for task, degree in enumerate(in_degree) if degree == 0])

    topological_order: list[int] = []

    while queue:
        vertex = queue.popleft()

        if vertex not in precedence_map or not precedence_map[vertex]:
            continue

        topological_order.append(vertex)

        for child in precedence_map[vertex]:
            in_degree[child] -= 1

            if in_degree[child] == 0:
                queue.append(child)

    return topological_order


def binary_search(
    array: Sequence[float],
    target: float,
    left: int = 0,
    right: int = -1,
) -> int:
    """
    Perform a binary search on a sorted array.

    Parameters
    ----------
    array: list
        The sorted array to be searched.

    target: Any
        The target value to be searched for.

    left: int, optional
        The left index of the search interval.

    right: int, optional
        The right index of the search interval.

    Returns
    -------
    int
        The index of the inclusion of the value in the array.
    """

    if right < 0:
        right = len(array) + right

    while left <= right:
        mid = (left + right) // 2

        if array[mid] == target:
            return mid

        if array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left

def infer_list_space(array: list[Any]) -> spaces.Space[Any]:
    "Infer the Gymnasium space for a list based on its elements."
    n = len(array)

    if n == 0:
        return spaces.Tuple([])

    elem = array[0]

    if isinstance(elem, str):
        return spaces.Tuple([spaces.Text(max_length=100) for _ in range(n)])

    if isinstance(elem, int):
        return spaces.Box(low=MIN_INT, high=MAX_INT, shape=(n,), dtype=np.int64)

    if isinstance(elem, float):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64)

    if isinstance(elem, bool):
        return spaces.MultiBinary(n)

    if is_iterable_type(array, str):
        return spaces.Tuple([spaces.Text(max_length=100) for _ in range(n)])

def scale_to_int(float_list: list[float], scale_factor: float = 1000.0) -> list[int]:
    "Scale a list of floats to integers using a common denominator."
    fractions = [Fraction(value).limit_denominator() for value in float_list]

    denominators = [fraction.denominator for fraction in fractions]
    lcm_denominator = lcm(*denominators)

    if lcm_denominator <= scale_factor:
        scale_factor = float(lcm_denominator)

    return [int(value * scale_factor) for value in float_list]

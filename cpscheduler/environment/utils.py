"""
    utils.py

    This module provides utility functions for the environment and other modules.
"""
from typing import Any, TypeVar, overload, SupportsInt
from collections.abc import Iterable, Mapping, Sequence
from typing_extensions import TypeIs

from collections import deque

import numpy as np
from gymnasium import spaces

from .common import MAX_INT, MIN_INT, TASK_ID

_S = TypeVar("_S")
_T = TypeVar("_T", bound=Any)
@overload
def convert_to_list(array: Any, dtype: type[_T]) -> list[_T]: ...

@overload
def convert_to_list(array: Iterable[_S], dtype: None = None) -> list[_S]: ...

@overload
def convert_to_list(array: Any, dtype: None = None) -> list[Any]: ...

def convert_to_list(array: Iterable[Any], dtype: type[_T] | None = None) -> list[Any]:
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
        for item in obj:
            if not isinstance(item, dtype):
                return False
        
        return True

    except Exception:
        return False

# Thank you mypy for not supporting Abstract classes with type[T]
def is_iterable_int(obj: Any) -> TypeIs[Iterable[SupportsInt]]:
    try:
        for item in obj:
            if not isinstance(item, SupportsInt):
                return False
        
        return True

    except Exception:
        return False

_K = TypeVar("_K")
_V = TypeVar("_V")
def is_mapping(
    obj: Mapping[Any, Any] | Any, keys_type: type[_K], values_type: type[_V]
) -> TypeIs[Mapping[_K, _V]]:
    """
    Returns whether the object is a mapping.

    Parameters
    ----------
    obj: Any
        The object to be checked.

    Returns
    -------
    bool
        Whether the object is a mapping.
    """
    if not isinstance(obj, Mapping):
        return False

    try:
        for key, value in obj.items():
            if not isinstance(key, keys_type) or not isinstance(value, values_type):
                return False

        return True

    except Exception:
        return False


def topological_sort(precedence_map: dict[TASK_ID, list[TASK_ID]], n_tasks: TASK_ID) -> list[TASK_ID]:
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
        return spaces.Box(low=MIN_INT, high=MAX_INT, shape=(n,), dtype=np.int64) # type: ignore

    if isinstance(elem, float):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64)

    if isinstance(elem, bool):
        return spaces.MultiBinary(n)

    if is_iterable_type(array, str):
        return spaces.Tuple([spaces.Text(max_length=100) for _ in range(n)])

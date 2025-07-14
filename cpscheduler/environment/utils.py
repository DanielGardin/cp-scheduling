"""
utils.py

This module provides utility functions for the environment and other modules.
"""

from typing import Any, TypeVar, overload
from collections.abc import Iterable, Sequence
from typing_extensions import TypeIs

from collections import deque

from ._common import TASK_ID, Int

_T = TypeVar("_T")


@overload
def convert_to_list(array: Any, dtype: type[_T]) -> list[_T]: ...
@overload
def convert_to_list(array: Iterable[_T], dtype: None = ...) -> list[_T]: ...
@overload
def convert_to_list(array: Any, dtype: None = ...) -> list[Any]: ...


def convert_to_list(array: Iterable[Any], dtype: type[Any] | None = None) -> list[Any]:
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


def is_iterable_type(
    obj: Any, dtype: type[_T], lazy: bool = True
) -> TypeIs[Iterable[_T]]:
    """
    Returns whether the object is an iterable containing elements of the specified type.

    Parameters
    ----------
    obj: Any
        The object to be checked.

    dtype: type
        The type of the elements of the iterable.

    lazy: bool, optional
        If True, the type checking will only check the first element of the iterable, the
        user is responsible for ensuring that all elements are of the specified type.
        In a compiled environment, the lazy check is preferred.

    Returns
    -------
    bool
        Whether the object is an iterable containing elements of the specified type.
    """
    try:
        if lazy:
            first_item = next(iter(obj))
            return isinstance(first_item, dtype)

        return all(isinstance(item, dtype) for item in obj)

    # except StopIteration:
    #     # If the iterable is empty, we consider it to be of the specified type
    #     return True

    except TypeError:
        # If the iterable is not a collection, it will raise a TypeError
        return False


def is_iterable_int(obj: Any, lazy: bool = True) -> TypeIs[Iterable[Int]]:
    try:
        if lazy:
            first_item = next(iter(obj))
            return isinstance(first_item, Int)

        return all(isinstance(item, Int) for item in obj)

    # except StopIteration:
    #     # If the iterable is empty, we consider it to be of the specified type
    #     return True

    except TypeError:
        # If the iterable is not a collection, it will raise a TypeError
        return False


def topological_sort(
    precedence_map: dict[TASK_ID, set[TASK_ID]], n_tasks: TASK_ID
) -> list[TASK_ID]:
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

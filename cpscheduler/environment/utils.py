from typing import (
    Any,
    TypeVar,
    Final,
    Optional,
    overload,
    Iterable,
    TypeGuard,
    Sequence,
)

import numpy as np
from collections import deque
from fractions import Fraction
from math import lcm

MIN_INT: Final[int] = -(2**24 + 1)
MAX_INT: Final[int] = 2**24 - 1

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
    Convert an iterable to a list. If a dtype is provided, the elements of the list will be casted to that type.

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


_Type = TypeVar("_Type", bound=object)


def is_iterable_type(obj: Any, dtype: type[_Type]) -> TypeGuard[Iterable[_Type]]:
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
        return isinstance(obj, Iterable) and all(
            [isinstance(item, dtype) for item in obj]
        )

    except Exception:  # If __iter__ is implemented but iterating raises an exception
        return False


_K = TypeVar("_K")
_V = TypeVar("_V")


def is_dict(
    obj: Any, keys_type: type[_K], values_type: type[_V]
) -> TypeGuard[dict[_K, _V]]:
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
    return isinstance(obj, dict) and all(
        [
            isinstance(key, keys_type) and isinstance(value, values_type)
            for key, value in obj.items()
        ]
    )


def get_elem_type(obj: Iterable[_Type]) -> type[_Type]:
    candidate = type(next(iter(obj)))

    for item in obj:
        if isinstance(item, candidate):
            continue

        other_type = type(item)
        if issubclass(candidate, other_type):
            candidate = other_type

        else:
            raise TypeError(
                f"Cannot determine a common type for the elements of the iterable: {candidate} and {other_type}"
            )

    return candidate


@overload
def invert(boolean: bool) -> bool: ...


@overload
def invert(boolean: list[bool]) -> list[bool]: ...


def invert(boolean: bool | list[bool]) -> bool | list[bool]:
    """
    Invert a boolean value or a list of boolean values.

    Parameters
    ----------
    boolean: bool or list[bool]
        The boolean value or list of boolean values to be inverted.

    Returns
    -------
    bool or list[bool]
        The inverted boolean value or list of boolean values.
    """

    if isinstance(boolean, bool):
        return not boolean

    return [not value for value in boolean]


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


from gymnasium import spaces


def infer_list_space(array: list[_T]) -> spaces.Space[Any]:
    n = len(array)

    if is_iterable_type(array, bool):
        return spaces.MultiBinary(n)

    if is_iterable_type(array, int):
        return spaces.Box(low=MIN_INT, high=MAX_INT, shape=(n,), dtype=np.int64)

    if is_iterable_type(array, float):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64)

    if is_iterable_type(array, str):
        return spaces.Tuple([spaces.Text(max_length=100) for _ in range(n)])

    raise TypeError(f"Cannot infer the space of the list: {array}")


def scale_to_int(float_list: list[float], scale_factor: float = 1000.0) -> list[int]:
    fractions = [Fraction(value).limit_denominator() for value in float_list]

    denominators = [fraction.denominator for fraction in fractions]
    lcm_denominator = lcm(*denominators)

    if lcm_denominator <= scale_factor:
        scale_factor = float(lcm_denominator)

    return [int(value * scale_factor) for value in float_list]

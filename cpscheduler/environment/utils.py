from typing import Any, TypeVar, Literal, Final, Optional, overload, Iterable, TypeGuard

from collections import deque

AVAILABLE_SOLVERS = Literal['cplex', 'ortools']

MIN_INT: Final[int] = -(2 ** 31 - 1)
MAX_INT: Final[int] =   2 ** 31 - 1


_S = TypeVar('_S')
_T = TypeVar('_T', bound=Any)
@overload
def convert_to_list(array: Any, dtype: type[_T]) -> list[_T]:
    ...

@overload
def convert_to_list(array: Iterable[_S], dtype: None = None) -> list[_S]:
    ...

@overload
def convert_to_list(array: Any, dtype: None = None) -> list[Any]:
    ...

def convert_to_list(array: Iterable[Any], dtype: Optional[type[_T]] = None) -> list[Any]:
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
            return list(array)

        return [dtype(item) for item in array]

    # If the iterable is not a collection, it will raise a TypeError
    except TypeError:
        return [array] if dtype is None else [dtype(array)]


def is_iterable_type(obj: Any, dtype: type[_T]) -> TypeGuard[Iterable[_T]]:
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
        return isinstance(obj, Iterable) and all([isinstance(item, dtype) for item in obj])

    except Exception: # If __iter__ is implemented but iterating raises an exception
        return False


@overload
def invert(boolean: bool) -> bool:
    ...

@overload
def invert(boolean: list[bool]) -> list[bool]:
    ...

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


def topological_sort(
        precedence_map: dict[int, list[int]],
        in_degree: list[int],
    ) -> list[int]:
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

    in_degree = in_degree.copy()

    queue = deque([task for task, degree in enumerate(in_degree) if degree == 0])

    topological_order: list[int] = []

    while queue:
        vertex = queue.popleft()

        children = precedence_map.get(vertex, [])

        if children:
            topological_order.append(vertex)

        for child in children:
            in_degree[child] -= 1

            if in_degree[child] == 0:
                queue.append(child)

    return topological_order


def binary_search(
        array: list[float],
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
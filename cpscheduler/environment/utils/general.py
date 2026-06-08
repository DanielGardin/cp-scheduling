"""General helper utilities for environment code."""

from collections.abc import Callable, Iterable
from typing import Any, TypeVar, overload

from cpscheduler.environment.utils.protocols import ArrayLike

_T = TypeVar("_T")

_IterableOrArray = Iterable[_T] | ArrayLike[_T]


@overload
def convert_to_list(
    array: _IterableOrArray[Any], dtype: type[_T]
) -> list[_T]: ...


@overload
def convert_to_list(
    array: _IterableOrArray[_T], dtype: None = ...
) -> list[_T]: ...


def convert_to_list(
    array: _IterableOrArray[Any], dtype: type[Any] | None = None
) -> list[Any]:
    """Convert an iterable to a list, optionally casting element types.

    Parameters
    ----------
    array : Iterable[Any]
        Iterable or ArrayLike to be converted to a list.

    dtype : type, optional
        Type used to cast each element in the list.

    Returns
    -------
    list[Any]
        List of elements from `array` (optionally cast to `dtype`).

    """
    if isinstance(array, ArrayLike):
        array = array.tolist()

    try:
        if dtype is None:
            return array if isinstance(array, list) else list(array)

        return [dtype(item) for item in array]

    # If the iterable is not a collection, it will raise a TypeError
    except TypeError:
        return [array] if dtype is None else [dtype(array)]


def extend_list(lst: list[_T], size: int, default: Callable[[], _T]) -> None:
    """Extend a list to `size` using values from `default`.

    Parameters
    ----------
    lst : list[_T]
        List to be extended in-place.

    size : int
        Target length for `lst`.

    default : Callable[[], _T]
        Factory used to produce new elements.

    """
    lst.extend([default() for _ in range(size - len(lst))])

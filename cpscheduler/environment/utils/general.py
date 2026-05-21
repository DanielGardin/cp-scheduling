from typing import Any, TypeVar, overload
from collections.abc import Iterable, Callable

_T = TypeVar("_T")


@overload
def convert_to_list(array: Iterable[Any], dtype: type[_T]) -> list[_T]: ...

@overload
def convert_to_list(array: Iterable[_T], dtype: None = ...) -> list[_T]: ...

def convert_to_list(
    array: Iterable[Any], dtype: type[Any] | None = None
) -> list[Any]:
    """
    Convert an iterable to a list. If a dtype is provided, the elements of the list will be casted
    to that type.

    Parameters
    ----------
    array: Iterable | None
        The iterable to be converted to a list.

    dtype: type, optional
        The type to which the elements of the list will be casted to.

    Returns
    -------
    list
        A list containing the elements of the iterable. If the array is None, initializes a new
        empty list of that given type.
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

def extend_list(lst: list[_T], size: int, default: Callable[[], _T]) -> None:
    lst.extend([default() for _ in range(size - len(lst))])

from typing import Any, TypeVar, SupportsInt
from collections.abc import Iterable
from typing_extensions import TypeIs


_T = TypeVar("_T")


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

    except StopIteration:
        # If the iterable is empty, we consider it to be of the specified type
        return True

    except TypeError:
        # If the iterable is not a collection, it will raise a TypeError
        return False


def is_iterable_int(obj: Any, lazy: bool = True) -> TypeIs[Iterable[SupportsInt | int]]:
    try:
        if lazy:
            first_item = next(iter(obj))
            return isinstance(first_item, (SupportsInt, int))

        return all(isinstance(item, (SupportsInt, int)) for item in obj)

    except TypeError:
        # If the iterable is not a collection, it will raise a TypeError
        return False


def iterate_indexed(obj: Iterable[Any]) -> list[dict[int, int]]:
    lst = []

    for item in obj:
        if isinstance(item, dict):
            lst.append({int(k): int(v) for k, v in item.items()})

        if is_iterable_int(item):
            lst.append({i: int(v) for i, v in enumerate(item)})

    return lst

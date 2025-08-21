from typing import Any, TypeVar, overload, Generic
from collections.abc import Iterator, Iterable, Sequence
from typing_extensions import TypeIs

import math
import random

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


# Maybe include JAX in another iteration?

from ._protocols import ArrayLike, TabularRepresentation

_T_co = TypeVar("_T_co", bound=Any, covariant=True)
_S = TypeVar("_S", bound=Any)


def is_pure_iterable(obj: Any) -> TypeIs[Iterable[Any]]:
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict))


class ListWrapper(Generic[_T_co]):
    def __init__(self, data: Iterable[_T_co]) -> None:
        self._data = list(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._data),)

    @property
    def dtype(self) -> Any:
        if self._data:
            return type(self._data[0])

        return Any

    def astype(self, dtype: type[_S] | Any) -> "ListWrapper[_S]":
        return ListWrapper(
            [x.astype(dtype) if hasattr(x, "astype") else dtype(x) for x in self._data]
        )

    def __array__(self, dtype: Any = None) -> ArrayLike:
        from numpy import array

        return array(self._data, dtype=dtype)

    @overload
    def __getitem__(self, index: int) -> _T_co: ...

    @overload
    def __getitem__(
        self, index: slice | Iterable[int] | Iterable[bool]
    ) -> "ListWrapper[_T_co]": ...

    def __getitem__(
        self, index: int | Iterable[int] | slice | Iterable[bool]
    ) -> "_T_co | ListWrapper[_T_co]":
        if isinstance(index, int):
            return self._data[index]

        if isinstance(index, slice):
            return ListWrapper(self._data[index])

        if isinstance(index, Iterable):
            if all(isinstance(i, bool) for i in index):
                return ListWrapper(
                    [self._data[i] for i, flag in enumerate(index) if flag]
                )

            return ListWrapper([self._data[i] for i in index if isinstance(i, int)])

    def __setitem__(
        self, index: int | Iterable[int] | slice | Iterable[bool], value: Any
    ) -> None:
        if isinstance(index, int):
            self._data[index] = value
            return

        if isinstance(index, slice):
            self._data[index] = value
            return

        if isinstance(index, Iterable):
            if all(isinstance(i, bool) for i in index):
                for i, flag in enumerate(index):
                    if flag:
                        self._data[i] = value
                return

            for i in index:
                if isinstance(i, int):
                    self._data[i] = value

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[_T_co]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"ListWrapper({self._data})"

    @overload
    def __add__(self, other: Iterable[_T_co] | _T_co, /) -> "ListWrapper[_T_co]": ...
    @overload
    def __add__(self, other: Any, /) -> "ListWrapper[Any]": ...

    def __add__(self, other: Any, /) -> Any:
        if is_pure_iterable(other):
            out = ListWrapper([x + y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in addition operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x + other for x in self._data])

    def __radd__(self, other: Any, /) -> "ListWrapper[Any]":
        return self.__add__(other)

    @overload
    def __sub__(self, other: Iterable[_T_co] | _T_co, /) -> "ListWrapper[_T_co]": ...
    @overload
    def __sub__(self, other: Any, /) -> "ListWrapper[Any]": ...

    def __sub__(self, other: Any, /) -> "ListWrapper[Any]":
        if is_pure_iterable(other):
            out = ListWrapper([x - y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in subtraction operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x - other for x in self._data])

    @overload
    def __rsub__(self, other: Iterable[_T_co] | _T_co, /) -> "ListWrapper[_T_co]": ...
    @overload
    def __rsub__(self, other: Any, /) -> "ListWrapper[Any]": ...

    def __rsub__(self, other: Any, /) -> "ListWrapper[Any]":
        if is_pure_iterable(other):
            out = ListWrapper([y - x for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in reverse subtraction operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([other - x for x in self._data])

    def __neg__(self) -> "ListWrapper[_T_co]":
        return ListWrapper([-x for x in self._data])

    @overload
    def __mul__(self, other: Iterable[_T_co] | _T_co, /) -> "ListWrapper[_T_co]": ...
    @overload
    def __mul__(self, other: Any, /) -> "ListWrapper[Any]": ...

    def __mul__(self, other: Any, /) -> "ListWrapper[Any]":
        if is_pure_iterable(other):
            out = ListWrapper([x * y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in multiplication operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x * other for x in self._data])

    def __rmul__(self, other: Any, /) -> "ListWrapper[Any]":
        return self.__mul__(other)

    def __truediv__(self, other: Any, /) -> "ListWrapper[Any]":
        if is_pure_iterable(other):
            out = ListWrapper([x / y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in division operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x / other for x in self._data])

    def __rtruediv__(self, other: Any, /) -> "ListWrapper[Any]":
        if is_pure_iterable(other):
            out = ListWrapper([y / x for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in division operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([other / x for x in self._data])

    def __eq__(self, other: Any, /) -> "ListWrapper[bool]":  # type: ignore
        if is_pure_iterable(other):
            out = ListWrapper([x == y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in equality operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x == other for x in self._data])

    def __ne__(self, other: Any, /) -> "ListWrapper[bool]":  # type: ignore
        if is_pure_iterable(other):
            out = ListWrapper([x != y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in inequality operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x != other for x in self._data])

    def __lt__(self, other: Any, /) -> "ListWrapper[bool]":
        if is_pure_iterable(other):
            out = ListWrapper([x < y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in less than operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x < other for x in self._data])

    def __le__(self, other: Any, /) -> "ListWrapper[bool]":
        if is_pure_iterable(other):
            out = ListWrapper([x <= y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in less than or equal operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x <= other for x in self._data])

    def __gt__(self, other: Any, /) -> "ListWrapper[bool]":
        if is_pure_iterable(other):
            out = ListWrapper([x > y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in greater than operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x > other for x in self._data])

    def __ge__(self, other: Any, /) -> "ListWrapper[bool]":
        if is_pure_iterable(other):
            out = ListWrapper([x >= y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in greater than or equal operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x >= other for x in self._data])

    def __contains__(self, item: Any, /) -> bool:
        return item in self._data

    def __bool__(self) -> bool:
        return bool(self._data)

    def __hash__(self) -> int:
        return hash(tuple(self._data))

    def max(self) -> _T_co:
        return max(self._data)

    def min(self) -> _T_co:
        return min(self._data)

    def sum(self) -> _T_co:
        sum_value = self._data[0]

        for x in self._data[1:]:
            sum_value += x

        return sum_value

    def mean(self) -> Any:
        if not self._data:
            return 0

        return self.sum() / len(self._data)

    @staticmethod
    def argsort(
        x: Iterable[_T_co], reverse: bool = False, stable: bool = False
    ) -> "ListWrapper[int]":
        if not stable:
            return ListWrapper(
                map(
                    lambda pair: pair[0],
                    sorted(
                        [(i, value) for i, value in enumerate(x)],
                        key=lambda pair: pair[1],
                        reverse=reverse,
                    ),
                )
            )

        indices = (
            [(value, i) for i, value in enumerate(x)]
            if not reverse
            else [(value, -i) for i, value in enumerate(x)]
        )

        indices.sort(reverse=reverse)
        argsort_lst = ListWrapper([i for _, i in indices])
        if reverse:
            argsort_lst = -argsort_lst

        return argsort_lst

    @staticmethod
    def maximum(
        x1: Iterable[_T_co] | _T_co,
        x2: Iterable[_T_co] | _T_co,
    ) -> "ListWrapper[_T_co]":
        if is_pure_iterable(x1):
            if is_pure_iterable(x2):
                return ListWrapper([a if a > b else b for a, b in zip(x1, x2)])

            else:
                return ListWrapper([a if a > x2 else x2 for a in x1])

        elif is_pure_iterable(x2):
            return ListWrapper([x1 if x1 > b else b for b in x2])

        return ListWrapper([max(x1, x2)])

    @staticmethod
    def minimum(
        x1: Iterable[_T_co] | _T_co,
        x2: Iterable[_T_co] | _T_co,
    ) -> "ListWrapper[_T_co]":
        if is_pure_iterable(x1):
            if is_pure_iterable(x2):
                return ListWrapper([a if a < b else b for a, b in zip(x1, x2)])

            else:
                return ListWrapper([a if a < x2 else x2 for a in x1])

        elif is_pure_iterable(x2):
            return ListWrapper([x1 if x1 < b else b for b in x2])

        return ListWrapper([min(x1, x2)])

    @staticmethod
    def log(x: Iterable[_T_co]) -> "ListWrapper[float]":
        return ListWrapper([math.log(a) for a in x])

    @staticmethod
    def exp(x: Iterable[_T_co]) -> "ListWrapper[float]":
        return ListWrapper([math.exp(a) for a in x])

    @staticmethod
    def sqrt(x: Iterable[_T_co]) -> "ListWrapper[float]":
        return ListWrapper([math.sqrt(a) for a in x])

    @staticmethod
    def where(
        condition: Iterable[bool],
        x1: Iterable[_T_co] | _T_co,
        x2: Iterable[_T_co] | _T_co,
    ) -> "ListWrapper[_T_co]":
        if is_pure_iterable(x1):
            if is_pure_iterable(x2):
                return ListWrapper(
                    [a if cond else b for cond, a, b in zip(condition, x1, x2)]
                )

            else:
                return ListWrapper(
                    [a if cond else x2 for cond, a in zip(condition, x1)]
                )

        elif is_pure_iterable(x2):
            return ListWrapper([x1 if cond else b for cond, b in zip(condition, x2)])

        return ListWrapper([x1 if cond else x2 for cond in condition])

    @staticmethod
    def sort(
        x: Iterable[_T_co], reverse: bool = False, stable: bool = False
    ) -> "ListWrapper[int]":
        if not stable:
            return ListWrapper(sorted(x, reverse=reverse))

        indices = (
            [(value, i) for i, value in enumerate(x)]
            if not reverse
            else [(value, -i) for i, value in enumerate(x)]
        )

        indices.sort(reverse=reverse)
        return ListWrapper([value for value, _ in indices])


maximum = ListWrapper.maximum
minimum = ListWrapper.minimum
log = ListWrapper.log
exp = ListWrapper.exp
sqrt = ListWrapper.sqrt
argsort = ListWrapper.argsort
sort = ListWrapper.sort

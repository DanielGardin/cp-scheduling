from typing import TypeVar, overload, Any, Generic
from collections.abc import Iterator, Iterable
from typing_extensions import TypeIs

import math

from contextlib import contextmanager

NUMPY_AVAILABLE = True
try:
    from numpy.typing import NDArray
    import numpy as np

except ImportError:
    NUMPY_AVAILABLE = False

TORCH_AVAILABLE = True
try:
    import torch

except ImportError:
    TORCH_AVAILABLE = False

from cpscheduler.environment._common import ObsType, Status
from ._protocols import ArrayLike, TabularRepresentation


@contextmanager
def disable_numpy() -> Iterator[None]:
    """
    Context manager to temporarily disable NumPy functionality in priority dispatching.
    This is mainly used for testing the ListWrapper functionality, without relying on NumPy.

    The reason for this is because we design the library to work with minimal dependencies,
    and NumPy is not a strict requirement for the core functionality.
    """
    global NUMPY_AVAILABLE
    original_numpy_available = NUMPY_AVAILABLE
    NUMPY_AVAILABLE = False
    try:
        yield

    finally:
        NUMPY_AVAILABLE = original_numpy_available


@contextmanager
def disable_torch() -> Iterator[None]:
    global TORCH_AVAILABLE
    original_torch_available = TORCH_AVAILABLE
    TORCH_AVAILABLE = False
    try:
        yield

    finally:
        TORCH_AVAILABLE = original_torch_available


def array_factory(data: Iterable[Any] | ArrayLike) -> ArrayLike:
    if NUMPY_AVAILABLE:
        return np.array(data)

    return ListWrapper(data)


def unify_obs(obs: ObsType) -> TabularRepresentation[ArrayLike]:
    tasks, jobs = obs

    new_obs: dict[str, ArrayLike] = {}

    for task_feature in tasks:
        new_obs[task_feature] = array_factory(tasks[task_feature])

    array_jobs_ids = new_obs["job_id"]
    for job_feature in jobs:
        if job_feature == "job_id":
            continue

        new_obs[job_feature] = array_factory(jobs[job_feature])[array_jobs_ids]

    return new_obs


def filter_tasks(
    obs: TabularRepresentation[ArrayLike], status: Status
) -> tuple[TabularRepresentation[ArrayLike], TabularRepresentation[ArrayLike]]:
    """
    Filters the tasks in the observation based on their status.
    """
    mask = obs[status] < Status.COMPLETED

    if isinstance(obs, dict):
        new_obs: TabularRepresentation[ArrayLike] = {}
        for k, v in obs.items():
            filtered_v = v[mask]

            assert isinstance(filtered_v, ArrayLike)
            new_obs[k] = filtered_v

        return new_obs, []

    if isinstance(obs, ArrayLike):
        new_obs = obs[obs[status] < Status.COMPLETED]

    raise TypeError(f"Unsupported observation type: {type(obs)}")


def wrap_observation(obs: Any) -> TabularRepresentation[ArrayLike]:
    "Wraps the observation into an ArrayLike structure."
    if isinstance(obs, tuple) and len(obs) == 2:
        assert all(isinstance(o, dict) for o in obs), "Expected a tuple of two dicts."
        return unify_obs(obs)

    if isinstance(obs, dict):
        return unify_obs((obs, {}))

    if isinstance(obs, ArrayLike):
        return obs

    raise TypeError(f"Couldnt wrap observation of type {type(obs)}.")


_T = TypeVar("_T", bound=Any, covariant=True)
_S = TypeVar("_S", bound=Any)


def is_pure_iterable(obj: Any) -> TypeIs[Iterable[Any]]:
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict))


class ListWrapper(Generic[_T]):
    def __init__(self, data: Iterable[_T]) -> None:
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
        if not NUMPY_AVAILABLE:
            raise RuntimeError(
                "Trying to convert ListWrapper to array in a non-NumPy environment."
            )

        return np.array(self._data, dtype=dtype)

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(
        self, index: slice | Iterable[int] | Iterable[bool]
    ) -> "ListWrapper[_T]": ...

    def __getitem__(
        self, index: int | Iterable[int] | slice | Iterable[bool]
    ) -> "_T | ListWrapper[_T]":
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

    def __iter__(self) -> Iterator[_T]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"ListWrapper({self._data})"

    @overload
    def __add__(self, other: Iterable[_T] | _T, /) -> "ListWrapper[_T]": ...
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
    def __sub__(self, other: Iterable[_T] | _T, /) -> "ListWrapper[_T]": ...
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
    def __rsub__(self, other: Iterable[_T] | _T, /) -> "ListWrapper[_T]": ...
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

    def __neg__(self) -> "ListWrapper[_T]":
        return ListWrapper([-x for x in self._data])

    @overload
    def __mul__(self, other: Iterable[_T] | _T, /) -> "ListWrapper[_T]": ...
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

    def sort(self, key: Any = None, reverse: bool = False) -> None:
        self._data.sort(key=key, reverse=reverse)

    def argsort(
        self, key: Any = None, reverse: bool = False, stable: bool = False
    ) -> "ListWrapper[int]":
        if not stable:
            return ListWrapper(
                sorted(
                    range(len(self._data)),
                    key=lambda i: self._data[i],
                    reverse=reverse,
                )
            )

        indices = (
            [(x, i) for i, x in enumerate(self._data)]
            if not reverse
            else [(x, -i) for i, x in enumerate(self._data)]
        )

        indices.sort(reverse=reverse)
        argsort = ListWrapper([i for _, i in indices])
        if reverse:
            argsort = -argsort
        return argsort

    def max(self) -> _T:
        return max(self._data)

    def min(self) -> _T:
        return min(self._data)

    def sum(self) -> _T:
        sum_value = self._data[0]

        for x in self._data[1:]:
            sum_value += x

        return sum_value

    def mean(self) -> Any:
        if not self._data:
            return 0

        return self.sum() / len(self._data)


def maximum(x1: Any, x2: Any) -> ArrayLike:
    if isinstance(x1, ListWrapper):
        if isinstance(x2, ListWrapper):
            if len(x1) != len(x2):
                raise ValueError(
                    f"Length mismatch in maximum operation. Expected {len(x1)}, got {len(x2)}"
                )

            return ListWrapper([a if a > b else b for a, b in zip(x1, x2)])

        else:
            return ListWrapper([a if a > x2 else x2 for a in x1])

    elif isinstance(x2, ListWrapper):
        return ListWrapper([x2 if a > x2 else a for a in x1])

    result: ArrayLike
    if (
        TORCH_AVAILABLE
        and isinstance(x1, torch.Tensor)
        and isinstance(x2, torch.Tensor)
    ):
        result = torch.maximum(x1, x2)

    elif NUMPY_AVAILABLE:
        result = np.maximum(x1, x2)

    else:
        raise TypeError(
            f"Unsupported types for maximum operation: {type(x1)}, {type(x2)}"
        )

    return result


def minimum(x1: Any, x2: Any) -> ArrayLike:
    if isinstance(x1, ListWrapper):
        if isinstance(x2, ListWrapper):
            if len(x1) != len(x2):
                raise ValueError(
                    f"Length mismatch in minimum operation. Expected {len(x1)}, got {len(x2)}"
                )

            return ListWrapper([a if a < b else b for a, b in zip(x1, x2)])

        return ListWrapper([a if a < x2 else x2 for a in x1])

    elif isinstance(x2, ListWrapper):
        return ListWrapper([x2 if a < x2 else a for a in x1])

    result: ArrayLike
    if (
        TORCH_AVAILABLE
        and isinstance(x1, torch.Tensor)
        and isinstance(x2, torch.Tensor)
    ):
        result = torch.minimum(x1, x2)

    elif NUMPY_AVAILABLE:
        result = np.minimum(x1, x2)

    else:
        raise TypeError(
            f"Unsupported types for minimum operation: {type(x1)}, {type(x2)}"
        )

    return result


def log(x: ArrayLike) -> ArrayLike:
    if isinstance(x, ListWrapper):
        return ListWrapper([math.log(a) for a in x])

    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = torch.log(x)

    elif NUMPY_AVAILABLE:
        result = np.log(x)

    else:
        raise TypeError(f"Unsupported types for log operation: {type(x)}")

    return result


def exp(x: ArrayLike) -> ArrayLike:
    if isinstance(x, ListWrapper):
        return ListWrapper([math.exp(a) for a in x])

    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = torch.exp(x)

    elif NUMPY_AVAILABLE:
        result = np.exp(x)

    else:
        raise TypeError(f"Unsupported types for log operation: {type(x)}")

    return result


def sqrt(x: ArrayLike) -> ArrayLike:
    if isinstance(x, ListWrapper):
        return ListWrapper([math.sqrt(a) for a in x])

    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = torch.sqrt(x)

    elif NUMPY_AVAILABLE:
        result = np.sqrt(x)

    elif hasattr(x, "sqrt"):
        result = x.sqrt()

    else:
        raise TypeError(f"Unsupported types for sqrt operation: {type(x)}")

    return result


def where(condition: ArrayLike, x1: Any, x2: Any) -> ArrayLike:
    if isinstance(condition, ListWrapper):
        if isinstance(x1, ListWrapper) and isinstance(x2, ListWrapper):
            return ListWrapper(
                [a if cond else b for cond, a, b in zip(condition, x1, x2)]
            )

        elif isinstance(x1, ListWrapper):
            return ListWrapper([a if cond else x2 for cond, a in zip(condition, x1)])

        elif isinstance(x2, ListWrapper):
            return ListWrapper([x1 if cond else b for cond, b in zip(condition, x2)])

        else:
            return ListWrapper([x1 if cond else x2 for cond in condition])

    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(condition, torch.Tensor):
        result = torch.where(condition, x1, x2)

    elif NUMPY_AVAILABLE:
        result = np.where(condition, x1, x2)

    else:
        raise TypeError(f"Unsupported types for where operation: {type(condition)}")

    return result


def argsort(x: ArrayLike, descending: bool = False, stable: bool = False) -> ArrayLike:
    if isinstance(x, ListWrapper):
        return x.argsort(reverse=descending, stable=stable)

    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = torch.argsort(x, descending=descending, stable=stable, dim=-1)

    elif NUMPY_AVAILABLE:
        result = np.argsort(x if not descending else -x, axis=-1, stable=stable)

    else:
        raise TypeError(f"Unsupported types for argsort operation: {type(x)}")

    return result

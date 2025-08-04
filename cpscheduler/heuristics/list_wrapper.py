from typing import TypeVar, overload, Any, Generic
from collections.abc import Iterator, Iterable
from typing_extensions import TypeIs

from contextlib import contextmanager

NUMPY_AVAILABLE = True
try:
    from numpy.typing import NDArray

except ImportError:
    NUMPY_AVAILABLE = False

from cpscheduler.environment._common import ObsType
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


def unify_obs(obs: ObsType) -> TabularRepresentation[ArrayLike]:
    tasks, jobs = obs

    new_obs: dict[str, ArrayLike] = {}

    if NUMPY_AVAILABLE:
        from numpy import array

        for task_feature in tasks:
            new_obs[task_feature] = array(tasks[task_feature])

        array_jobs_ids = new_obs["job_id"]

        for job_feature in jobs:
            if job_feature == "job_id":
                continue

            new_obs[job_feature] = array(jobs[job_feature])[array_jobs_ids]

    else:
        for task_feature in tasks:
            new_obs[task_feature] = ListWrapper(tasks[task_feature])

        list_jobs_ids: Iterable[int] = tasks["job_id"]

        for job_feature in jobs:
            if job_feature == "job_id":
                continue

            new_obs[job_feature] = ListWrapper(jobs[job_feature])[list_jobs_ids]

    return new_obs

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


_T = TypeVar('_T', bound=Any, covariant=True)
_S = TypeVar('_S', bound=Any)

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

    def astype(self, dtype: type[_S] | Any) -> 'ListWrapper[_S]':
        return ListWrapper(
            [x.astype(dtype) if hasattr(x, 'astype') else dtype(x) for x in self._data]
        )

    if NUMPY_AVAILABLE:
        def __array__(self, dtype: Any = None) -> ArrayLike:
            from numpy import array

            return array(self._data, dtype=dtype)

    else:
        def __array__(self, dtype: Any = None) -> ArrayLike:
            return self.astype(dtype) if dtype else self

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice | Iterable[int] | Iterable[bool]) -> 'ListWrapper[_T]': ...

    def __getitem__(self, index: int | Iterable[int] | slice | Iterable[bool]) -> '_T | ListWrapper[_T]':
        if isinstance(index, int):
            return self._data[index]
        
        if isinstance(index, slice):
            return ListWrapper(self._data[index])
        
        if isinstance(index, Iterable):
            if all(isinstance(i, bool) for i in index):
                return ListWrapper([self._data[i] for i, flag in enumerate(index) if flag])

            return ListWrapper([self._data[i] for i in index if isinstance(i, int)])

    def __setitem__(self, index: int | Iterable[int] | slice | Iterable[bool], value: Any) -> None:
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
    def __add__(self, other: Iterable[_T] | _T, /) -> 'ListWrapper[_T]': ...
    @overload
    def __add__(self, other: Any, /) -> 'ListWrapper[Any]': ...

    def __add__(self, other: Any, /) -> Any:
        if is_pure_iterable(other):
            out = ListWrapper([x + y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in addition operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x + other for x in self._data])

    @overload
    def __sub__(self, other: Iterable[_T] | _T, /) -> 'ListWrapper[_T]': ...
    @overload
    def __sub__(self, other: Any, /) -> 'ListWrapper[Any]': ...

    def __sub__(self, other: Any, /) -> 'ListWrapper[Any]':
        if is_pure_iterable(other):
            out = ListWrapper([x - y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in subtraction operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x - other for x in self._data])

    def __neg__(self) -> 'ListWrapper[_T]':
        return ListWrapper([-x for x in self._data])

    @overload
    def __mul__(self, other: Iterable[_T] | _T, /) -> 'ListWrapper[_T]': ...
    @overload
    def __mul__(self, other: Any, /) -> 'ListWrapper[Any]': ...

    def __mul__(self, other: Any, /) -> 'ListWrapper[Any]':
        if is_pure_iterable(other):
            out = ListWrapper([x * y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in multiplication operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x * other for x in self._data])

    def __rmul__(self, other: Any, /) -> 'ListWrapper[Any]':
        return self.__mul__(other)

    def __truediv__(self, other: Any, /) -> 'ListWrapper[Any]':
        if is_pure_iterable(other):
            out = ListWrapper([x / y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in division operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x / other for x in self._data])

    def __eq__(self, other: Any, /) -> 'ListWrapper[bool]': # type: ignore
        if is_pure_iterable(other):
            out = ListWrapper([x == y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in equality operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x == other for x in self._data])
    
    def __ne__(self, other: Any, /) -> 'ListWrapper[bool]': # type: ignore
        if is_pure_iterable(other):
            out = ListWrapper([x != y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in inequality operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x != other for x in self._data])
    
    def __lt__(self, other: Any, /) -> 'ListWrapper[bool]':
        if is_pure_iterable(other):
            out = ListWrapper([x < y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in less than operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x < other for x in self._data])
    
    def __le__(self, other: Any, /) -> 'ListWrapper[bool]':
        if is_pure_iterable(other):
            out = ListWrapper([x <= y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in less than or equal operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x <= other for x in self._data])
    
    def __gt__(self, other: Any, /) -> 'ListWrapper[bool]':
        if is_pure_iterable(other):
            out = ListWrapper([x > y for x, y in zip(self._data, other)])
            if len(out) != len(self):
                raise ValueError(
                    f"Length mismatch in greater than operation. Expected {len(self)}, got {len(out)}"
                )

            return out

        return ListWrapper([x > other for x in self._data])
    
    def __ge__(self, other: Any, /) -> 'ListWrapper[bool]':
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

    def argsort(self, key: Any = None, reverse: bool = False) -> 'ListWrapper[int]':        
        return ListWrapper(
            sorted(range(len(self._data)), key=lambda i: self._data[i], reverse=reverse)
        )



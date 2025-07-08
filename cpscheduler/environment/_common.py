"Common types and constants used in the environment module."

from typing import (
    Any,
    Final,
    TypeAlias,
    SupportsInt,
    SupportsFloat,
    Hashable,
    Iterator,
    Protocol,
    runtime_checkable,
)
from collections.abc import Iterable, Mapping

from mypy_extensions import i64, i32, i16, u8

# Type aliases for commonly used types and also for performance optimization with mypyc
MACHINE_ID: TypeAlias = i16
TASK_ID: TypeAlias = i32
PART_ID: TypeAlias = i16
TIME: TypeAlias = i32

Int: TypeAlias = SupportsInt | int | i64 | i32 | i16 | u8
Float: TypeAlias = SupportsFloat | float | i64 | i32 | i16 | u8

# Reducing upper bounds to avoid numerical issues
MIN_INT: Final[TIME] = -(2**24 + 1)
MAX_INT: Final[TIME] = 2**24 - 1

# Allowed types for task and job data
ScalarType: TypeAlias = bool | int | float | str


@runtime_checkable
class DataFrameLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def columns(self) -> Any: ...

    def __getitem__(self, key: Hashable) -> Any: ...

    def __iter__(self) -> Iterator[Hashable]: ...


ProcessTimeAllowedTypes: TypeAlias = (
    Iterable[Mapping[SupportsInt, SupportsInt]]
    | Iterable[Iterable[SupportsInt]]
    | Iterable[SupportsInt]  # Requires a machine array
    | str  # Requires a machine array
    | Iterable[str]  # Map columns in data to machines
    | None  # Infer from data
)

InstanceTypes: TypeAlias = DataFrameLike | Mapping[Hashable, Iterable[Any]]

ObsType: TypeAlias = tuple[dict[str, list[Any]], dict[str, list[Any]]]
InfoType: TypeAlias = dict[str, Any]

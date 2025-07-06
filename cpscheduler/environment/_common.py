"Common types and constants used in the environment module."

from typing import (
    Any,
    Final,
    TypeAlias,
    SupportsInt,
    Sequence,
    Protocol,
    runtime_checkable,
)
from collections.abc import Iterable, Mapping

from mypy_extensions import i16, i32, u8

# Type aliases for commonly used types and also for performance optimization with mypyc
MACHINE_ID: TypeAlias = i16
TASK_ID: TypeAlias = i32
PART_ID: TypeAlias = i16
TIME: TypeAlias = i32

# Reducing upper bounds to avoid numerical issues
MIN_INT: Final[TIME] = -(2**24 + 1)
MAX_INT: Final[TIME] = 2**24 - 1

# Allowed types for task and job data
ScalarType: TypeAlias = bool | int | float | str

@runtime_checkable
class DataFrameLike(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def columns(self) -> Any: ...

    def __getitem__(self, key: str) -> Any: ...

    def head(self, n: int = ...) -> Any: ...

    def to_dict(self) -> Mapping[str, Sequence[Any]]: ...


ProcessTimeAllowedTypes: TypeAlias = (
    Iterable[Mapping[SupportsInt, SupportsInt]]
    | Iterable[Iterable[SupportsInt]]
    | Iterable[SupportsInt]  # Requires a machine array
    | str  # Requires a machine array
    | Iterable[str]  # Map columns in data to machines
    | None  # Infer from data
)

InstanceTypes: TypeAlias = DataFrameLike | Mapping[str, Iterable[Any]]

ObsType: TypeAlias = tuple[dict[str, list[Any]], dict[str, list[Any]]]
InfoType: TypeAlias = dict[str, Any]

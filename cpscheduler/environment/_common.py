"Common types and constants used in the environment module."

from typing import (
    Any,
    Final,
    SupportsInt,
    SupportsFloat,
    Protocol,
    runtime_checkable,
)
from collections.abc import Iterable, Mapping, Hashable, Iterator
from typing_extensions import TypedDict, TypeAlias

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


@runtime_checkable
class DataFrameLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def columns(self) -> Any: ...

    def __getitem__(self, key: Hashable) -> Any: ...

    def __iter__(self) -> Iterator[Hashable]: ...


InstanceTypes: TypeAlias = DataFrameLike | Mapping[Any, Iterable[Any]]

ObsType: TypeAlias = tuple[dict[str, list[Any]], dict[str, list[Any]]]

InfoType: TypeAlias = dict[str, Any]


class InstanceConfig(TypedDict, total=False):
    "Configuration for loading an instance."

    instance: InstanceTypes


Options: TypeAlias = dict[str, Any] | InstanceConfig | None


class Status:
    "Possible statuses of a task at a given time."

    # awaiting:  time < start_lb[0] or waiting for a machine
    AWAITING: Final[u8] = 0

    # paused:    start_lb[i] + duration[i] < = time < start_lb[i+1] for some i
    PAUSED: Final[u8] = 1

    # executing: start_lb[i] <= time < start_lb[i] + duration[i] for some i
    EXECUTING: Final[u8] = 2

    # completed: time >= start_lb[-1] + duration[-1]
    COMPLETED: Final[u8] = 3


def ceil_div(a: TIME, b: TIME) -> TIME:
    "a divided by b, rounded up to the nearest integer."
    return -(-a // b)

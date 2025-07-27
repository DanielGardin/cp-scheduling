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
from collections.abc import Iterable, Mapping, Sequence
from typing_extensions import NotRequired, TypedDict, TypeAlias

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


# Changing from Int to Any to avoid issues with Mypy in nested generics (explore later)
ProcessTimeAllowedTypes: TypeAlias = (
    # DataFrameLike | # TODO: Handle DataFrame-like structures in schedule_setup.py
    Iterable[Mapping[Any, Any]]
    | Iterable[Sequence[Any]]
    | Iterable[Int]  # Requires a machine array
    | str  # Requires a machine array
    | Iterable[str]  # Map columns in data to machines
    | None  # Infer from data
)

InstanceTypes: TypeAlias = DataFrameLike | Mapping[Any, Iterable[Any]]
MachineDataTypes: TypeAlias = DataFrameLike | Mapping[Any, Iterable[Any]]

ObsType: TypeAlias = tuple[dict[str, list[Any]], dict[str, list[Any]]]

InfoType: TypeAlias = dict[str, Any]

# TODO: With the addition of metrics, this might need to be updated
# class InfoType(TypedDict, total=False):
#     "Type for the info dictionary in the environment."

#     n_queries: int
#     current_time: int


class InstanceConfig(TypedDict):
    "Instance configuration for the environment."

    instance: NotRequired[InstanceTypes]
    processing_times: NotRequired[ProcessTimeAllowedTypes]
    job_instance: NotRequired[InstanceTypes]
    job_feature: NotRequired[str]
    machine_instance: NotRequired[InstanceTypes]


class EnvSerialization(TypedDict):
    "Serialization format for the environment."

    setup: dict[str, Any]
    constraints: dict[str, dict[str, Any]]
    objective: dict[str, Any]

    instance: NotRequired[InstanceConfig]


class Status:
    "Possible statuses of a task at a given time."

    # awaiting:  time < start_lb[0] or waiting for a machine
    AWAITING: Final[u8] = 0

    # available: time < start_lb[0] and can be executed
    AVAILABLE: Final[u8] = 1

    # executing: start_lb[i] <= time < start_lb[i] + duration[i] for some i
    EXECUTING: Final[u8] = 2

    # paused:    start_lb[i] + duration[i] < = time < start_lb[i+1] for some i
    PAUSED: Final[u8] = 3

    # completed: time >= start_lb[-1] + duration[-1]
    COMPLETED: Final[u8] = 4

    # unknown status
    UNKNOWN: Final[u8] = 5

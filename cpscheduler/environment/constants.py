"Common types and constants used in the environment module."

from typing import TypeAlias, Final, SupportsInt, SupportsFloat

from mypy_extensions import i64, i32, i16, u8


# ------------------------------------------------------------------------------
# Type aliases for commonly used types

IndexType: TypeAlias = i32

MachineID: TypeAlias = IndexType
TaskID: TypeAlias = IndexType

Time: TypeAlias = i32

# Generic numeric types
Int: TypeAlias = SupportsInt | int | i64 | i32 | i16 | u8
Float: TypeAlias = SupportsFloat | float | i64 | i32 | i16 | u8

# ------------------------------------------------------------------------------
# Constants

MIN_TIME: Final[Time] = 0
MAX_TIME: Final[Time] = (1 << 31) - 1

# Special machine ID representing a global machine
GLOBAL_MACHINE_ID: MachineID = -1

# ------------------------------------------------------------------------------
# Enums

StatusType: TypeAlias = u8


class Status:
    "Possible statuses of a task at a given time."

    __slots__ = ()

    AWAITING: Final[StatusType] = 0
    "Task is awaiting execution, typically when time <= start_lb."

    PAUSED: Final[StatusType] = 1
    "Task has been started, but has been paused and now is waiting to be resumed."

    EXECUTING: Final[StatusType] = 2
    "Task is currently executing on a machine."

    COMPLETED: Final[StatusType] = 3
    "Task has been completed and is no longer active in the schedule."

    INFEASIBLE: Final[StatusType] = 255
    "Task is not consistent with the constraints and cannot be scheduled."

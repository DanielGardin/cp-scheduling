"""Core types and constants for the environment package."""

from typing import (
    Final,
    SupportsFloat,
    SupportsInt,
)

from mypy_extensions import i32

# ------------------------------------------------------------------------------
# Type aliases for commonly used types

IndexType = i32

MachineID = IndexType
TaskID = IndexType
JobID = IndexType

Time = i32

# Generic numeric types
# Altought it seems redundant to union int and SupportsInt, for some reason,
# mypy does not consider its own integer types (u8, i16, i32, i64) as subclasses
# of SupportsInt.
Int = SupportsInt | int
Float = SupportsFloat | float

# ------------------------------------------------------------------------------
# Constants

MIN_TIME: Final[Time] = 0
MAX_TIME: Final[Time] = (1 << 31) - 1

# Sentinel constants.
GLOBAL_MACHINE_ID: MachineID = -1
UNKNOWN_TASK: TaskID = -1
TIMELESS: Time = -1

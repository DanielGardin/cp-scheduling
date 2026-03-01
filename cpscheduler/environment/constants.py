"Common types and constants used in the environment module."

from typing import Final, SupportsInt, SupportsFloat
from typing_extensions import TypeAlias

from mypy_extensions import i64, i32, i16, u8


# ------------------------------------------------------------------------------
# Type aliases for commonly used types

# Currently MACHINE_ID and TASK_ID must have the same type due to indexing
INDEX_TYPE: TypeAlias = i32
MACHINE_ID: TypeAlias = INDEX_TYPE
TASK_ID: TypeAlias = INDEX_TYPE

TIME: TypeAlias = i32

# Generic numeric types
Int: TypeAlias = SupportsInt | int | i64 | i32 | i16 | u8
Float: TypeAlias = SupportsFloat | float | i64 | i32 | i16 | u8

# ------------------------------------------------------------------------------
# Constants

MIN_TIME: Final[TIME] = 0
MAX_TIME: Final[TIME] = (1 << 31) - 1

# Special machine ID representing a global machine
GLOBAL_MACHINE_ID: MACHINE_ID = -1

# ------------------------------------------------------------------------------
# Enums

StatusType: TypeAlias = u8

class Status:
    "Possible statuses of a task at a given time."

    # awaiting:  time < start_lb[0] or waiting for a machine
    AWAITING: Final[StatusType] = 0

    # paused:    start_lb[i] + duration[i] < = time < start_lb[i+1] for some i
    PAUSED: Final[StatusType] = 1

    # executing: start_lb[i] <= time < start_lb[i] + duration[i] for some i
    EXECUTING: Final[StatusType] = 2

    # completed: time >= start_lb[-1] + duration[-1]
    COMPLETED: Final[StatusType] = 3

    # unfeasible: task cannot be completed given the current state
    INFEASIBLE: Final[StatusType] = 255


PresenceType: TypeAlias = u8

class Presence:
    UNDEFINED: Final[PresenceType] = 0

    PRESENT: Final[PresenceType] = 1

    ABSENT: Final[PresenceType] = 2

    INFEASIBLE: Final[PresenceType] = 3

    FIXED: Final[PresenceType] = 4
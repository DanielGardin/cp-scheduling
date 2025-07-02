"Common types and constants used in the environment module."
from typing import Final, TypeAlias, SupportsInt
from collections.abc import Iterable, Mapping

from mypy_extensions import i16, i32, u8

MACHINE_ID: TypeAlias = i16
TASK_ID:    TypeAlias = i32
PART_ID:    TypeAlias = i16
TIME:       TypeAlias = i32

# Reducing upper bounds to avoid numerical issues
MIN_INT: Final[TIME] = -(2**24 + 1)
MAX_INT: Final[TIME] = 2**24 - 1

ProcessTimeAllowedTypes: TypeAlias = (
    Iterable[Mapping[SupportsInt, SupportsInt]] |
    Iterable[Iterable[SupportsInt]]     |
    Iterable[SupportsInt]               | # Requires a machine array
    str                         | # Requires a machine array
    Iterable[str]               | # Map columns in data to machines
    None                          # Infer from data
)
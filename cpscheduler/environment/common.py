"Common types and constants used in the environment module."
from typing import Final, TypeAlias, Iterable, Mapping

# Reducing upper bounds to avoid numerical issues
MIN_INT: Final[int] = -(2**24 + 1)
MAX_INT: Final[int] = 2**24 - 1

ProcessTimeAllowedTypes: TypeAlias = (
    Iterable[Mapping[int, int]] |
    Iterable[Iterable[int]]     |
    Iterable[int]               | # Requires a machine array
    str                         | # Requires a machine array
    Iterable[str]               | # Map columns in data to machines
    None                          # Infer from data
)
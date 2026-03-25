from typing import Any, Literal, TypeAlias, overload
from collections.abc import Iterable, Mapping, Sequence
from typing_extensions import TypedDict, NotRequired

from datetime import timedelta

import minizinc


class SolverConfig(TypedDict, total=False):
    processes: NotRequired[int]
    timeout: NotRequired[timedelta | None]
    optimisation_level: NotRequired[int]
    free_search: NotRequired[bool]
    random_seed: NotRequired[int | None]

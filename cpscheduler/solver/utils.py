from typing import Any, Literal, overload, TypeVar
from collections.abc import Coroutine, Iterable
from datetime import timedelta

from fractions import Fraction
from math import lcm

import asyncio

TimeUnits = Literal["ms", "s", "m", "h", "d"]

@overload
def resolve_timeout(timeout: int, timeout_unit: TimeUnits) -> timedelta: ...

@overload
def resolve_timeout(timeout: None, timeout_unit: TimeUnits) -> None: ...

def resolve_timeout(
    timeout: int | None,
    timeout_unit: TimeUnits,
) -> timedelta | None:
    if timeout is None:
        return None

    if timeout_unit == "ms":
        return timedelta(milliseconds=timeout)

    if timeout_unit == "s":
        return timedelta(seconds=timeout)

    if timeout_unit == "m":
        return timedelta(minutes=timeout)

    if timeout_unit == "h":
        return timedelta(hours=timeout)

    if timeout_unit == "d":
        return timedelta(days=timeout)

_T = TypeVar("_T")
def run_couroutine(coro: Coroutine[Any, Any, _T]) -> _T:
    loop = asyncio.get_event_loop()

    # Ew
    if loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()

    return asyncio.run(coro)

def scale_to_int(
    coefficients: Iterable[float], constant: float = 1.0, max_scale: int = 1000
) -> tuple[list[int], int]:
    "Scale a list of floats to integers using a common denominator."
    float_list = [*coefficients, constant]

    fractions = [Fraction(value).limit_denominator() for value in float_list]

    lcm_denominator = lcm(*(fraction.denominator for fraction in fractions))

    scale = lcm_denominator if lcm_denominator <= max_scale else max_scale

    scaled_list = [int(value * scale) for value in float_list]

    return scaled_list[:-1], scaled_list[-1]

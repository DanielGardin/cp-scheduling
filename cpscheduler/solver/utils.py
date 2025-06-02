from typing import Any, Literal, Optional, overload, TypeVar, Coroutine
from datetime import timedelta

import asyncio

TimeUnits = Literal['ms', 's', 'm', 'h', 'd']

@overload
def resolve_timeout(timeout: int, timeout_unit: TimeUnits) -> timedelta:
    ...

@overload
def resolve_timeout(timeout: None, timeout_unit: TimeUnits) -> None:
    ...

def resolve_timeout(
    timeout: Optional[int],
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

    raise ValueError(f"Time unit {timeout_unit} not recognized. Use 'ms', 's', 'm', 'h', or 'd'.")

_T = TypeVar('_T')
def run_couroutine(coro: Coroutine[Any, Any, _T]) -> _T:
    loop = asyncio.get_event_loop()

    # Ew
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()

    return asyncio.run(coro)
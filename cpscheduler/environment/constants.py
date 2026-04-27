"Common types and constants used in the environment module."

from typing import (
    Any, Final, SupportsInt, SupportsFloat, Literal, ClassVar, cast, final
)
from collections.abc import Iterable, Mapping, Hashable
from typing_extensions import Self

from mypy_extensions import i64, i32, i16, u8, mypyc_attr

# ------------------------------------------------------------------------------
# Type aliases for commonly used types

IndexType = i32

MachineID = IndexType
TaskID = IndexType

Time = i32

# Generic numeric types
Int = SupportsInt | int | i64 | i32 | i16 | u8
Float = SupportsFloat | float

# ------------------------------------------------------------------------------
# Constants

MIN_TIME: Final[Time] = 0
MAX_TIME: Final[Time] = (1 << 31) - 1

# Special machine ID representing a global machine
GLOBAL_MACHINE_ID: MachineID = -1

# ------------------------------------------------------------------------------
# Enums

@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class Enum:
    def __init__(self) -> None:
        raise ValueError(
            f"Cannot instantiate enum class {type(self).__name__}"
        )


StatusType = Literal[0, 1, 2, 3]


class Status(Enum):
    "Possible statuses of a task at a given time."

    AWAITING: Final[Literal[0]] = 0
    "Task is awaiting execution, typically when time <= start_lb."

    PAUSED: Final[Literal[1]] = 1
    "Task has been started, but has been paused and now is waiting to be resumed."

    EXECUTING: Final[Literal[2]] = 2
    "Task is currently executing on a machine."

    COMPLETED: Final[Literal[3]] = 3
    "Task has been completed and is no longer active in the schedule."

# ------------------------------------------------------------------------------
# Pickling utils

PickleState = list[tuple[str, Any]]
"Internal state of an object when serializing it"


def _to_hashable(value: Any) -> Any:
    "Convert nested containers into a hashable representation."

    if isinstance(value, Mapping):
        return frozenset(
            (_to_hashable(key), _to_hashable(val))
            for key, val in value.items()
        )

    if isinstance(value, set | frozenset):
        return frozenset(_to_hashable(item) for item in value)

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(_to_hashable(item) for item in value)

    if isinstance(value, Hashable):
        return value

    raise TypeError(f"Value of type {type(value).__name__} is not hashable")

def _collect_fields(cls: type) -> tuple[str, ...]:
    fields = getattr(cls, "__ez_fields__", None)
    if fields is not None:
        return cast(tuple[str, ...], fields)

    # mypyc path (authoritative)
    attrs = getattr(cls, "__mypyc_attrs__", None)
    if attrs is not None:
        return tuple(
            name for name in cast(tuple[str, ...], attrs)
            if not (name.startswith("__") and name.endswith("__"))
        )

    # interpreted fallback: __annotations__ only
    seen: set[str] = set()
    result: list[str] = []

    for c in reversed(cls.__mro__):
        annotations = cast(
            dict[str, type], c.__dict__.get("__annotations__", {})
        )

        for name in annotations:
            if (name.startswith("__") and name.endswith("__")):
                continue

            if name not in seen:
                seen.add(name)
                result.append(name)

    return tuple(result)

def _iter_state(obj: object) -> tuple[str, ...]:
    cls = type(obj)
    return tuple(
        name for name in _collect_fields(cls)
        if hasattr(obj, name)
    )

@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class EzPickle:
    """
    mypyc-compatible structural serialization.

    Field source:
    - compiled: __mypyc_attrs__
    - interpreted: __annotations__
    """
    __args__: ClassVar[tuple[str, ...] | None] = None

    @final
    def __getstate__(self) -> PickleState:
        return [(name, getattr(self, name)) for name in _iter_state(self)]

    @final
    def __setstate__(self, state: PickleState | dict[str, Any]) -> None:
        items = state.items() if isinstance(state, dict) else state
        for key, value in items:
            object.__setattr__(self, key, value)

    @final
    def __reduce__(self) -> tuple[type[Self], tuple[Any, ...], dict[str, Any]] | tuple[type[Self], tuple[()], PickleState]:
        cls = type(self)
        fields = _collect_fields(cls)
        args_spec = cls.__args__

        if args_spec is not None:
            args = tuple(getattr(self, k) for k in args_spec)
            state = {
                k: getattr(self, k)
                for k in fields
                if k not in args_spec and hasattr(self, k)
            }
            return (cls, args, state)

        return (cls, (), self.__getstate__())

    def __repr__(self) -> str:
        cls = type(self)
        parts = [
            f"{name}={getattr(self, name)!r}"
            for name in _collect_fields(cls)
            if not name.startswith("_") and hasattr(self, name)
        ]

        return f"{cls.__name__}({', '.join(parts)})"

    def __hash__(self) -> int:
        return hash(_to_hashable(self.__getstate__()))

    # FUTURE: Not sure why mypyC can't compile EzPickle with __eq__
    # but when we remove the implementation, everything works fine.
    # def __eq__(self, value: Any) -> bool:
    #     if not isinstance(value, EzPickle):
    #         return NotImplemented
        
    #     return self.__getstate__() == value.__getstate__()

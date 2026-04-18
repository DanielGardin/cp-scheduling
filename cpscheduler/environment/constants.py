"Common types and constants used in the environment module."

from typing import (
    Any, TypeAlias, Final, SupportsInt, SupportsFloat, Literal, final
)
from typing_extensions import Self
from collections.abc import Iterator

from mypy_extensions import i64, i32, i16, u8

import copy

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

class Enum:
    __slots__ = ()

    def __init__(self) -> None:
        raise ValueError(f"Cannot instantiate enum class {self.__class__}")


StatusType: TypeAlias = Literal[0, 1, 2, 3]


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

PickleState: TypeAlias = list[tuple[str, Any]]
"Internal state of an object when serializing it"

ReduceReturn: TypeAlias = tuple[type, PickleState, PickleState]

def _iter_slots(obj: object) -> Iterator[Any]:
    return (
        slot
        for cls in type(obj).__mro__
        for slot in getattr(cls, '__slots__', ())
        if slot != '__weakref__' and hasattr(obj, slot)
    )


class EzPickle:
    """
    Simplified EzPickle for classes with only optional/keyword parameters.
    
    You must implement __slots__ with the class-defined variables in order
    to correctly pickle/deepcopy the object.
    """

    __slots__ = ()

    @final
    def __reduce__(self) -> tuple[type, tuple[()], PickleState]:
        return (self.__class__, (), self.__getstate__())

    @final
    def __getstate__(self) -> PickleState:
        return [
            (slot, getattr(self, slot))
            for slot in _iter_slots(self)
        ]

    @final
    def __setstate__(self, state: PickleState) -> None:
        for key, value in state:
            object.__setattr__(self, key, value)

    @final
    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        cls = type(self)
        new_obj = cls.__new__(cls)

        memo[id(self)] = new_obj
        for slot in _iter_slots(self):
            object.__setattr__(
                new_obj, slot, copy.deepcopy(getattr(self, slot), memo)
            )

        return new_obj

    @final
    def __copy__(self) -> Self:
        cls = type(self)
        new_obj = cls.__new__(cls)

        for slot in _iter_slots(self):
            object.__setattr__(new_obj, slot, getattr(self, slot))

        return new_obj

    def __hash__(self) -> int:
        return hash(tuple(v for _, v in self.__getstate__()))

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, type(self)):
            return NotImplemented

        return self.__getstate__() == value.__getstate__()

    def __repr__(self) -> str:
        attributes = [
            f'{name}={obj!r}'
            for name, obj in self.__getstate__()
            if not name.startswith('_')
        ]

        return f"{type(self).__name__}({', '.join(attributes)})"



class CustomDataclass:
    __slots__ = ()

    @final
    def __reduce__(self) -> tuple[type, tuple[()], dict[str, Any]]:
        return (self.__class__, (), dict(self.__getstate__()))

    @final
    def __getstate__(self) -> PickleState:
        return [
            (slot, getattr(self, slot))
            for slot in _iter_slots(self)
        ]

    @final
    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        cls = type(self)
        new_obj = cls.__new__(cls)

        memo[id(self)] = new_obj
        for slot in _iter_slots(self):
            object.__setattr__(
                new_obj, slot, copy.deepcopy(getattr(self, slot), memo)
            )

        return new_obj

    @final
    def __copy__(self) -> Self:
        cls = type(self)
        new_obj = cls.__new__(cls)

        for slot in _iter_slots(self):
            object.__setattr__(new_obj, slot, getattr(self, slot))

        return new_obj

    def __hash__(self) -> int:
        return hash(tuple(v for _, v in self.__getstate__()))

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, type(self)):
            return NotImplemented

        return self.__getstate__() == value.__getstate__()

    def __repr__(self) -> str:
        attributes = [
            f'{name}={obj!r}'
            for name, obj in self.__getstate__()
            if not name.startswith('_')
        ]

        return f"{type(self).__name__}({', '.join(attributes)})"
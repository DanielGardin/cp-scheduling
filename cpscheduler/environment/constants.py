"Common types and constants used in the environment module."

from collections.abc import Hashable, Iterable, Mapping
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    Self,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    cast,
    final,
)

from mypy_extensions import i32, mypyc_attr

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

# Special machine ID representing a global machine
GLOBAL_MACHINE_ID: MachineID = -1

# ------------------------------------------------------------------------------
# Enums


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class Enum:
    __enum_count__: ClassVar[int] = 0

    def __new__(cls) -> Self:
        raise TypeError(f"Cannot instantiate enum class {cls.__name__}")

    def __init_subclass__(cls) -> None:
        cls.__enum_count__ = sum(
            1
            for k in vars(cls)
            if not (k.startswith("__") and k.endswith("__"))
        )

    @classmethod
    def count(cls) -> int:
        return cls.__enum_count__


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
# Singletons


class Singleton:
    _created: ClassVar[bool] = False

    def __new__(cls) -> Self:
        if cls._created:
            raise ValueError(
                f"Singleton class {cls.__name__} can only be instantiated once."
            )

        instance = super().__new__(cls)
        cls._created = True
        return instance

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __bool__(self) -> bool:
        return False

    def __hash__(self) -> int:
        return hash(type(self))

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self


# ------------------------------------------------------------------------------
# Pickling utils

PickleState = list[tuple[str, Any]]
"Internal state of an object when serializing it"


def _to_hashable(value: Any) -> Any:
    "Convert nested containers into a hashable representation."

    if isinstance(value, Mapping):
        return frozenset(
            (_to_hashable(key), _to_hashable(val)) for key, val in value.items()
        )

    if isinstance(value, set | frozenset):
        return frozenset(_to_hashable(item) for item in value)

    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
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
            name
            for name in cast(tuple[str, ...], attrs)
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
            if name.startswith("__") and name.endswith("__"):
                continue

            if name not in seen:
                seen.add(name)
                result.append(name)

    return tuple(result)


def _iter_state(obj: object) -> tuple[str, ...]:
    cls = type(obj)
    return tuple(name for name in _collect_fields(cls) if hasattr(obj, name))


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class EzPickle:
    """
    mypyc-compatible structural serialization.

    Field source:
    - compiled: __mypyc_attrs__
    - interpreted: __annotations__
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        return super().__new__(cls)

    @final
    def __getstate__(self) -> PickleState:
        return [(name, getattr(self, name)) for name in _iter_state(self)]

    @final
    def __setstate__(self, state: PickleState | dict[str, Any]) -> None:
        items = state.items() if isinstance(state, dict) else state
        for key, value in items:
            object.__setattr__(self, key, value)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        cls = type(self)

        return (
            cls.__new__,
            (cls,),
            self.__getstate__(),
        )

    def __repr__(self) -> str:
        cls = type(self)
        parts = [
            f"{name}={getattr(self, name)!r}"
            for name in _collect_fields(cls)
            if not name.startswith("_") and hasattr(self, name)
        ]

        return f"{cls.__name__}({', '.join(parts)})"

    # FUTURE: Not sure why mypyC can't compile EzPickle with __eq__
    # but when we remove the implementation, everything works fine.
    # def __eq__(self, value: Any) -> bool:
    #     return (
    #         isinstance(value, EzPickle)
    #         and self.__getstate__() == value.__getstate__()
    #     )

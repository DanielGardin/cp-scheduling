"""Core types, constants, and utility base classes for the environment package.

This module defines:

- Semantic type aliases for scheduling entities and time values.
- Shared environment constants.
- Execution-state enumerations.
- Serialization and singleton utility base classes.

"""

from collections.abc import Hashable, Iterable, Mapping
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    cast,
    final,
)

from mypy_extensions import i32, mypyc_attr
from typing_extensions import Self

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

# Sentinel machine identifier representing non-machine-specific operations.
GLOBAL_MACHINE_ID: MachineID = -1

# ------------------------------------------------------------------------------
# Enums


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class Enum:
    """Lightweight namespace-style enumeration base class.

    Subclasses define enumeration members as class attributes. The total
    number of declared members is computed automatically during subclass
    creation.

    This implementation is intended for low-overhead use in mypyc-compiled
    code and does not provide the full semantics of :class:`enum.Enum`.

    Methods
    -------
    count() -> int
        Return the number of enumeration members defined on the subclass.

    Examples
    --------
    >>> class MyEnum(Enum):
    ...     FIRST = 0
    ...     SECOND = 1
    >>> MyEnum.count()
    2

    Notes
    -----
    - Enumeration classes cannot be instantiated.
    - Any non-dunder class attribute contributes to the enumeration count.

    """

    __enum_count__: ClassVar[int] = 0

    def __new__(cls) -> Self:
        """Prevent instantiation of enum classes.

        Enums are not meant to be instantiated, they serve as namespaces for
        constant values.
        """
        raise TypeError(f"Cannot instantiate enum class {cls.__name__}")

    def __init_subclass__(cls) -> None:
        """Compute the number of enumeration members defined by the subclass."""
        cls.__enum_count__ = sum(
            1
            for k in vars(cls)
            if not (k.startswith("__") and k.endswith("__"))
        )

    @classmethod
    def count(cls) -> int:
        """Return the total number of enumeration values defined in the subclass."""
        return cls.__enum_count__


StatusType = Literal[0, 1, 2, 3]


class Status(Enum):
    """Enumeration of task execution states.

    Tasks transition between execution states during schedule evaluation:

    ``AWAITING -> EXECUTING -> COMPLETED``

    If preemption is enabled, tasks may additionally transition through
    ``PAUSED`` states before completion.

    Attributes
    ----------
    AWAITING : Literal[0]
        Task is eligible for future execution but is not currently running.

    PAUSED : Literal[1]
        Task execution has been temporarily suspended.

    EXECUTING : Literal[2]
        Task is actively executing on an assigned machine.

    COMPLETED : Literal[3]
        Task execution has finished.

    Notes
    -----
    - Enumeration values are stable integer constants.
    - ``PAUSED`` is only relevant for preemptive scheduling models.

    """

    AWAITING: Final[Literal[0]] = 0
    """Task is awaiting execution, typically when time <= start_lb."""

    PAUSED: Final[Literal[1]] = 1
    """Task has been started, but has been paused and now is waiting to be resumed."""

    EXECUTING: Final[Literal[2]] = 2
    """Task is currently executing on a machine."""

    COMPLETED: Final[Literal[3]] = 3
    """Task has been completed and is no longer active in the schedule."""


# ------------------------------------------------------------------------------
# Singletons


class Singleton:
    """Base class enforcing unique-instance semantics.

    Each subclass may be instantiated at most once during program execution.
    Additional instantiation attempts raise :class:`ValueError`.

    Intended primarily for sentinel objects and globally unique markers.

    Notes
    -----
    - Singleton instances evaluate to ``False`` in boolean contexts.
    - Copy and deepcopy operations preserve identity.
    - Instance creation is not thread-safe.

    Examples
    --------
    >>> class Missing(Singleton):
    ...     pass
    >>> x = Missing()
    >>> Missing()
    Traceback (most recent call last):
        ...
    ValueError

    """

    _created: ClassVar[bool] = False

    def __new__(cls) -> Self:
        """Allow exactly one instance of each Singleton subclass. Subsequent calls raise ValueError."""
        if cls._created:
            raise ValueError(
                f"Singleton class {cls.__name__} can only be instantiated once."
            )

        instance = super().__new__(cls)
        cls._created = True
        return instance

    def __repr__(self) -> str:
        """Return a simple string representation of the singleton instance."""
        return f"{type(self).__name__}()"

    def __bool__(self) -> bool:
        """Singleton instances evaluate to False in boolean context."""
        return False

    def __hash__(self) -> int:
        """Return a unique hash for the singleton instance based on its type."""
        return hash(type(self))

    def __copy__(self) -> Self:
        """When copying a singleton, the same instance is returned."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Deep copying a singleton returns the same instance."""
        return self


# ------------------------------------------------------------------------------
# Pickling utils

# Serialized object state represented as (field_name, value) pairs.
PickleState = list[tuple[str, Any]]


def _to_hashable(value: Any) -> Any:
    """Convert nested containers into a hashable representation.

    Recursively transforms mutable containers (list, dict, set) into their immutable counterparts
    (tuple, frozenset) for use as dictionary keys or in sets. Strings and bytes are treated as
    already hashable and returned as-is.

    Parameters
    ----------
    value : Any
        The value to convert. Can be a nested structure of containers.

    Returns
    -------
    Any
        Hashable representation of the input. Mutable containers are converted; hashable values
        are returned unchanged.

    Raises
    ------
    TypeError
        If value is an unhashable type not handled by this function (e.g., custom class without __hash__).

    Examples
    --------
    >>> _to_hashable({'a': [1, 2], 'b': {3, 4}})
    frozenset({('a', (1, 2)), ('b', frozenset({3, 4}))})

    >>> _to_hashable([1, 2, 3])
    (1, 2, 3)

    >>> _to_hashable({1, 2, 3})
    frozenset({1, 2, 3})

    """
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
    """Collect serializable field names for a class.

    Field discovery follows the priority order:

    1. ``__ez_fields__``
    2. ``__mypyc_attrs__``
    3. ``__annotations__`` collected across the MRO

    Parameters
    ----------
    cls : type
        Class to inspect.

    Returns
    -------
    tuple[str, ...]
        Ordered tuple of non-dunder field names.

    Notes
    -----
    - Inherited annotated fields preserve MRO order.
    - Used internally by :class:`EzPickle`.

    """
    fields = getattr(cls, "__ez_fields__", None)
    if fields is not None:
        return cast("tuple[str, ...]", fields)

    # mypyc path (authoritative)
    attrs = getattr(cls, "__mypyc_attrs__", None)
    if attrs is not None:
        return tuple(
            name
            for name in cast("tuple[str, ...]", attrs)
            if not (name.startswith("__") and name.endswith("__"))
        )

    # interpreted fallback: __annotations__ only
    seen: set[str] = set()
    result: list[str] = []

    for c in reversed(cls.__mro__):
        annotations = cast(
            "dict[str, type]", c.__dict__.get("__annotations__", {})
        )

        for name in annotations:
            if name.startswith("__") and name.endswith("__"):
                continue

            if name not in seen:
                seen.add(name)
                result.append(name)

    return tuple(result)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class EzPickle:
    """Automatic pickle support for mypyc-compatible classes.

    Object state is serialized as a sequence of ``(field_name, value)``
    pairs derived from annotated or explicitly registered fields.

    Supports both interpreted Python classes and mypyc-compiled classes.

    Examples
    --------
    >>> class Point(EzPickle):
    ...     x: int
    ...     y: int
    ...
    ...     def __init__(self, x: int, y: int):
    ...         self.x = x
    ...         self.y = y

    >>> import pickle
    >>> p = Point(1, 2)
    >>> q = pickle.loads(pickle.dumps(p))
    >>> (q.x, q.y)
    (1, 2)

    Notes
    -----
    - Fields are discovered from annotations or mypyc metadata.
    - Private fields are serialized but omitted from ``__repr__``.
    - Lazily initialized attributes are serialized only if present.

    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Create an uninitialized instance for pickle reconstruction.

        This override ensures compatibility with mypyc-generated constructors
        during unpickling.
        """
        return super().__new__(cls)

    @final
    def __getstate__(self) -> PickleState:
        """Collect the state of the object for pickling."""
        return [
            (name, getattr(self, name))
            for name in _collect_fields(type(self))
            if hasattr(self, name)
        ]

    @final
    def __setstate__(self, state: PickleState | dict[str, Any]) -> None:
        """Restore the state of the object from the pickled state."""
        items = state.items() if isinstance(state, dict) else state
        for key, value in items:
            object.__setattr__(self, key, value)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        """Return the pickle reduction tuple for the instance."""
        cls = type(self)

        return (
            cls.__new__,
            (cls,),
            self.__getstate__(),
        )

    def __repr__(self) -> str:
        """Return a repr containing public field values."""
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

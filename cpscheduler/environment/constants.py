"""Core types, constants, and utility base classes for the environment package.

This module defines:

- Semantic type aliases for scheduling entities and time values.
- Shared environment constants.
- Execution-state enumerations.
- Serialization and singleton utility base classes.

"""

import hashlib
from inspect import get_annotations
from typing import (
    Any,
    ClassVar,
    Final,
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


def _frame(b: bytes) -> bytes:
    return len(b).to_bytes(8, byteorder="big") + b


def _canonical_bytes(obj: Any) -> bytes:
    if obj is None:
        return b"N"
    if isinstance(obj, bool):
        return b"b1" if obj else b"b0"
    if isinstance(obj, int):
        return b"i" + str(obj).encode()
    if isinstance(obj, float):
        return b"f" + repr(obj).encode()
    if isinstance(obj, str):
        return b"s" + obj.encode()
    if isinstance(obj, bytes):
        return b"y" + obj

    if isinstance(obj, dict):
        items = sorted(
            _frame(_canonical_bytes(k)) + _frame(_canonical_bytes(v))
            for k, v in obj.items()
        )
        return b"d" + b"".join(items)

    if isinstance(obj, (list, tuple)):
        tag = b"l" if isinstance(obj, list) else b"t"
        return tag + b"".join(_frame(_canonical_bytes(item)) for item in obj)

    if isinstance(obj, (set, frozenset)):
        items = sorted(_frame(_canonical_bytes(item)) for item in obj)
        return b"e" + b"".join(items)

    if isinstance(obj, EzPickle):
        return b"z" + _frame(_canonical_bytes(sorted(obj.__getstate__())))

    return b"r" + repr(obj).encode("utf-8")


def hash_anything(obj: Any) -> int:
    """Compute a hash for any object, including nested containers.

    Parameters
    ----------
    obj : Any
        The object to convert.

    Returns
    -------
    int
        The hash of the transformed object, suitable for use in sets or as
        dictionary keys.

    Raises
    ------
    TypeError
        If the object is an unhashable type not handled by this function
        (e.g., custom class without __hash__).

    """
    digest = hashlib.sha256(_canonical_bytes(obj)).digest()
    return int.from_bytes(digest[:8], "big")


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
        annotations = get_annotations(c).keys()

        for name in annotations:
            if name.startswith("__") and name.endswith("__"):
                continue

            if name not in seen:
                seen.add(name)
                result.append(name)

    return tuple(result)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True, acyclic=True)
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

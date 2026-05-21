from typing import Any, Literal, Generic

from typing_extensions import Self, TypeVar

from cpscheduler.environment.constants import EzPickle

SemanticType = Literal[
    # Numerical
    "continuous",   # Real number
    "discrete",     # Integer number
    "binary",       # Boolean
    "count",        # Non-negative integers
    "cost",         # Non-negative real
    "probability",  # Real [0,1], interpreted as p(.)
    "normalized",   # Real [0,1]

    # Identity / indexing
    "id",           # ID Integer
    "order",        # Ordinal Integer
    "categorical",  # Arbitrary categories

    # Scheduling
    "task",         # Integer [0, n_tasks)
    "job",          # Integer [0, n_jobs)
    "machine",      # Integer [0, n_machines)
    "time",         # Integer [0, MAX_TIME)
    "duration",     # Integer delta [0, MAX_TIME)
    "interval",     # Interval (start, end)
    "calendar",     # Sequence of intervals (start, end)

    # Structural
    "mask",         # Boolean mask
    "set",          # Arbitrary set
    "sequence",     # Arbitrary sequence

    # Graph
    "adjacency",    # Adjacency matrix

    # Unknown
    "unknown",      # Non-structured data, ignored when serialized
]

Scope = Literal[
    "task",
    "job",
    "machine",
    "global",
]

_T = TypeVar('_T', default=Any)

class FeatureSpec(EzPickle, Generic[_T]):

    __args__ = ("scope", "semantic", "optional")

    scope: Scope
    semantic: SemanticType
    optional: bool

    def __init__(
        self,
        scope: Scope,
        semantic: SemanticType,
        optional: bool = False
    ) -> None:
        self.scope = scope
        self.semantic = semantic
        self.optional = optional

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, FeatureSpec)
            and self.scope == value.scope
            and self.semantic == value.semantic
            and self.optional == value.optional
        )

class UnsetType:
    def __repr__(self) -> str:
        return "UNSET"

UNSET = UnsetType()
"""Defines a Feature without default data value (unitialized by default)

Usually used to define a consumer, that will be filled by the user-specified 
instance later.
"""

class Feature(EzPickle, Generic[_T]):
    name: str
    spec: FeatureSpec[_T]
    _default: _T | UnsetType
    _data: _T

    _loaded: bool

    def __init__(
        self,
        name: str,
        scope: Scope,
        semantic: SemanticType,
        optional: bool = False,
        default: _T | UnsetType = UNSET
    ) -> None:
        self.name = name
        self.spec = FeatureSpec(
            scope,
            semantic,
            optional,
        )

        self._default = default

        self._loaded = False
        if not isinstance(default, UnsetType):
            self._data = default
            self._loaded = True

    @classmethod
    def from_spec(
        cls, name: str, spec: FeatureSpec[_T], default: _T | UnsetType = UNSET
    ) -> Self:
        return cls(
            name,
            spec.scope,
            spec.semantic,
            spec.optional,
            default
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def value(self) -> _T:
        return self._data

    def set_data(self, data: _T) -> None:
        if self._loaded:
            raise RuntimeError(
                f"Feature {self.name} already has loaded data: {self._data}."
            )

        self._data = data
        self._loaded = True

    def shared_data(self, source: "Feature[_T]") -> None:
        if self.spec != source.spec:
            raise ValueError(
                f"Source feature '{source.name}' has different specs: expected {self.spec}, "
                f"got {source.spec}."
            )
        
        self.set_data(source._data)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Feature)
            and self.name == value.name
            and self.spec == value.spec
        )

class TaskFeature(Feature[list[_T]]):

    elem_type: type[_T]

    def __init__(
        self,
        name: str,
        elem_type: type[_T],
        semantic: SemanticType,
        optional: bool = False,
        default: list[_T] | UnsetType = UNSET
    ) -> None:
        super().__init__(
            name=name,
            scope="task",
            semantic=semantic,
            optional=optional,
            default=default,
        )

        self.elem_type = elem_type

class JobFeature(Feature[list[_T]]):

    elem_type: type[_T]

    def __init__(
        self,
        name: str,
        elem_type: type[_T],
        semantic: SemanticType,
        optional: bool = False,
        default: list[_T] | UnsetType = UNSET
    ) -> None:
        super().__init__(
            name=name,
            scope="job",
            semantic=semantic,
            optional=optional,
            default=default,
        )

        self.elem_type = elem_type

class MachineFeature(Feature[list[_T]]):

    elem_type: type[_T]

    def __init__(
        self,
        name: str,
        elem_type: type[_T],
        semantic: SemanticType,
        optional: bool = False,
        default: list[_T] | UnsetType = UNSET
    ) -> None:
        super().__init__(
            name=name,
            scope="machine",
            semantic=semantic,
            optional=optional,
            default=default,
        )

        self.elem_type = elem_type

class GlobalFeature(Feature[_T]):

    pytype: type[_T]

    def __init__(
        self,
        name: str,
        pytype: type[_T],
        semantic: SemanticType,
        optional: bool = False,
        default: _T | UnsetType = UNSET
    ) -> None:
        super().__init__(
            name=name,
            scope="task",
            semantic=semantic,
            optional=optional,
            default=default,
        )

        self.pytype = pytype

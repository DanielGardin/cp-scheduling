from typing import Any, Generic

from typing_extensions import Self, TypeVar

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.specs import (
    FeatureSpec,
    Scope,
    ShapeDim,
    SemanticType,
)


class UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()
"""Defines a Feature without default data value (unitialized by default)

Usually used to define a consumer, that will be filled by the user-specified
instance later.
"""

_T = TypeVar("_T", default=Any)


class Feature(EzPickle, Generic[_T]):
    name: str
    spec: FeatureSpec
    _default: _T | UnsetType
    _data: _T

    _loaded: bool

    def __init__(
        self,
        name: str,
        scope: Scope,
        semantic: SemanticType,
        optional: bool = False,
        default: _T | UnsetType = UNSET,
        *,
        shape: tuple[ShapeDim, ...] = (),
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        self.name = name
        self.spec = FeatureSpec(
            scope=scope,
            semantic=semantic,
            optional=optional,
            shape=shape,
            n_categories=n_categories,
            low=low,
            high=high,
        )

        self._default = default

        self._loaded = False
        if not isinstance(default, UnsetType):
            self._data = default
            self._loaded = True

    @classmethod
    def from_spec(
        cls, name: str, spec: FeatureSpec, default: _T | UnsetType = UNSET
    ) -> Self:
        return cls(
            name=name,
            scope=spec.scope,
            semantic=spec.semantic,
            optional=spec.optional,
            default=default,
            shape=spec.shape,
            n_categories=spec.n_categories,
            low=spec.low,
            high=spec.high,
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


class TaskFeature(Feature[list[_T]]):

    elem_type: type[_T]

    def __init__(
        self,
        name: str,
        elem_type: type[_T],
        semantic: SemanticType,
        optional: bool = False,
        default: list[_T] | UnsetType = UNSET,
        *,
        shape: tuple[ShapeDim, ...] = (),
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        super().__init__(
            name=name,
            scope="task",
            semantic=semantic,
            optional=optional,
            default=default,
            shape=("n_tasks", *shape),
            n_categories=n_categories,
            low=low,
            high=high,
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
        default: list[_T] | UnsetType = UNSET,
        *,
        shape: tuple[ShapeDim, ...] = (),
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        super().__init__(
            name=name,
            scope="job",
            semantic=semantic,
            optional=optional,
            default=default,
            shape=("n_jobs", *shape),
            n_categories=n_categories,
            low=low,
            high=high,
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
        default: list[_T] | UnsetType = UNSET,
        *,
        shape: tuple[ShapeDim, ...] = (),
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        super().__init__(
            name=name,
            scope="machine",
            semantic=semantic,
            optional=optional,
            default=default,
            shape=("n_machines", *shape),
            n_categories=n_categories,
            low=low,
            high=high,
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
        default: _T | UnsetType = UNSET,
        *,
        shape: tuple[ShapeDim, ...] = (),
        n_categories: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            scope="global",
            semantic=semantic,
            optional=optional,
            default=default,
            shape=shape,
            n_categories=n_categories,
        )

        self.pytype = pytype

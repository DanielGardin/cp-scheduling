from collections.abc import Sequence
from typing import Any, Generic

from typing_extensions import Self, TypeVar

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.specs.feature_spec import (
    FeatureSpec,
    Scope,
    SemanticType,
)
from cpscheduler.environment.specs.symbols import (
    BaseShapeDim,
    ShapeDim,
)


class UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()
"""Defines a Feature without default data value (unitialized by default)

Usually used to define a consumer, that will be filled by the user-specified
instance later.
"""


def infer_shape(data: Any) -> tuple[int, ...]:
    if hasattr(data, "shape"):
        return tuple(data.shape)

    if isinstance(data, Sequence) and not isinstance(data, str):
        if len(data) == 0:
            return (0,)

        elem_shape = infer_shape(data[0])
        for elem in data[1:]:
            _shape = infer_shape(elem)

            if _shape != elem_shape:
                raise ValueError(
                    f"Inconsistent shapes in sequence: expected {elem_shape}, "
                    f"got {_shape}."
                )

        return (len(data), *elem_shape)

    return ()


def compare_shapes(
    inferred_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> bool:
    """Check inferred_shape == target_shape, under some conditions.

    Inferred shape cannot infer the complete shape when a dimension with zero
    elements is reached.
    For that reason, naively comparing inferred_shape == target_shape would
    trigger false negatives in empty sequences.

    To avoid such behavior, this function acknowledges this limitation and
    early returns True when
    - inferred_shape == target_shape, or
    - inferred_shape[:t] == target_shape[:t] and inferred_shape[t] = 0

    This is a limitation due to using python builtin containers.
    """

    for inf_dim, tgt_dim in zip(inferred_shape, target_shape, strict=False):
        if inf_dim == 0:
            return True

        if inf_dim != tgt_dim:
            return False

    return len(inferred_shape) == len(target_shape)


_T = TypeVar("_T", default=Any)


class Feature(EzPickle, Generic[_T]):
    name: str
    spec: FeatureSpec
    _default: _T | UnsetType
    _data: _T
    _observed_shape: tuple[int, ...] | None

    _loaded: bool
    dynamic: bool

    def __init__(
        self,
        name: str,
        scope: Scope,
        semantic: SemanticType,
        optional: bool = False,
        default: _T | UnsetType = UNSET,
        dynamic: bool = False,
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
        self.dynamic = dynamic
        self._observed_shape = None
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
        assert self._loaded
        return self._data

    @property
    def shape(self) -> tuple[BaseShapeDim, ...]:
        if self._observed_shape is not None:
            return self._observed_shape

        return self.spec.raw_shape

    def set_data(self, data: _T) -> None:
        if not self.dynamic and self._loaded:
            raise RuntimeError(
                f"Feature {self.name} already has loaded data: {self._data}."
            )

        self._data = data
        self._loaded = True
        self._observed_shape = infer_shape(data)

    def shared_data(self, source: "Feature[_T]") -> None:
        if not source.spec.shareable_with(self.spec):
            raise ValueError(
                f"Source feature '{source.name}' has different specs: "
                f"expected {self.spec}, got {source.spec}."
            )

        self.set_data(source._data)

    def validate(self, **symbol_values: int) -> None:
        if not self._loaded:
            if not self.spec.optional:
                raise ValueError(
                    f"Feature {self.name} is required but has no loaded data."
                )

            return

        if self._observed_shape is not None:
            target_shape = self.spec.resolve_shape(**symbol_values)

            if not compare_shapes(self._observed_shape, target_shape):
                raise ValueError(
                    f"Feature {self.name} has invalid shape: "
                    f"expected {target_shape}, got {self._observed_shape}."
                )

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Feature)
            and self.name == value.name
            and self.spec == value.spec
            and self._loaded == value._loaded
            and self._observed_shape == value._observed_shape
        )


class TaskFeature(Feature[list[_T]]):
    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        optional: bool = False,
        default: list[_T] | UnsetType = UNSET,
        dynamic: bool = False,
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
            dynamic=dynamic,
            shape=("n_tasks", *shape),
            n_categories=n_categories,
            low=low,
            high=high,
        )


class JobFeature(Feature[list[_T]]):
    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        optional: bool = False,
        default: list[_T] | UnsetType = UNSET,
        dynamic: bool = False,
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
            dynamic=dynamic,
            shape=("n_jobs", *shape),
            n_categories=n_categories,
            low=low,
            high=high,
        )


class MachineFeature(Feature[list[_T]]):
    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        optional: bool = False,
        default: list[_T] | UnsetType = UNSET,
        dynamic: bool = False,
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
            dynamic=dynamic,
            shape=("n_machines", *shape),
            n_categories=n_categories,
            low=low,
            high=high,
        )


class GlobalFeature(Feature[_T]):
    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        optional: bool = False,
        default: _T | UnsetType = UNSET,
        dynamic: bool = False,
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
            dynamic=dynamic,
            shape=shape,
            n_categories=n_categories,
        )

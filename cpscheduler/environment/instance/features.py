from collections.abc import Sequence
from typing import Any, Generic, Literal

from typing_extensions import Self, TypeIs, TypeVar

from cpscheduler.environment.constants import EzPickle, Singleton
from cpscheduler.environment.specs.feature_spec import (
    FeatureSpec,
    Scope,
    SemanticType,
)
from cpscheduler.environment.specs.symbols import (
    BaseShapeDim,
)


class _UnsetType(Singleton):
    """Defines a Feature without default data value (unitialized by default)

    Usually used to define a consumer, that will be filled by the user-specified
    instance later.
    """


UNSET = _UnsetType()


def is_unset(value: object) -> TypeIs[_UnsetType]:
    return value is UNSET


def has_shape(data: Any, shape: tuple[int | None, ...]) -> bool:
    if hasattr(data, "shape"):
        return tuple(data.shape) == shape

    if shape and shape[0] is None:
        return True

    if isinstance(data, Sequence) and not isinstance(data, str):
        if len(data) == 0:
            return True

        if len(shape) == 0:
            return False

        if len(data) != shape[0]:
            return False

        return has_shape(data[0], shape[1:])

    return len(shape) == 0


_T = TypeVar("_T", default=Any)


class Feature(EzPickle, Generic[_T]):
    name: str
    spec: FeatureSpec
    optional: bool
    owner: bool

    _default: _T | _UnsetType
    _data: _T | _UnsetType

    dynamic: bool

    def __init__(
        self,
        name: str,
        scope: Scope,
        semantic: SemanticType,
        *,
        optional: bool = False,
        default: _T | _UnsetType = UNSET,
        owner: bool | None = None,
        dynamic: bool = False,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        owns_data = not is_unset(default)

        if owner is None:
            self.owner = owns_data

        elif not owner and owns_data:
            raise ValueError(
                f"Feature '{name}' is explicitly not an owner, but provides "
                "default data. A non-provider must have the default data unset."
            )

        else:
            self.owner = owner

        self.optional = optional
        self.name = name
        self.spec = FeatureSpec(
            scope=scope,
            semantic=semantic,
            shape=shape,
            n_categories=n_categories,
            low=low,
            high=high,
        )

        self._default = default

        self.dynamic = dynamic
        self._data = default

    @classmethod
    def from_spec(
        cls,
        name: str,
        spec: FeatureSpec,
        *,
        optional: bool = False,
        default: _T | _UnsetType = UNSET,
    ) -> Self:
        return cls(
            name=name,
            scope=spec.scope,
            semantic=spec.semantic,
            optional=optional,
            default=default,
            shape=spec.raw_shape,
            n_categories=spec.n_categories,
            low=spec.low,
            high=spec.high,
        )

    @property
    def loaded(self) -> bool:
        return self._data is not UNSET

    @property
    def value(self) -> _T:
        if not is_unset(self._data):
            return self._data

        raise ValueError(f"Feature {self.name} has no loaded data.")

    @property
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return self.spec.raw_shape

    def reset(self) -> None:
        default = self._default

        self._data = default

    def own_data(self, data: _T) -> None:
        """Tells the instance that the data being loaded is owned by this
        feature.

        Use it when you need to set data, but want to avoid being overwritten
        by user instance data.
        """
        if self.owner:
            raise ValueError(
                f"Feature '{self.name}' already is an owner, use `set_data` instead."
            )

        self.owner = True
        self._data = data

    def set_data(self, data: _T) -> None:
        if not self.dynamic and self.loaded:
            raise RuntimeError(
                f"Feature '{self.name}' already has loaded data: {self._data}."
            )

        self._data = data

    def shared_data(self, source: "Feature[_T]") -> None:
        if not source.spec.shareable_with(self.spec):
            raise ValueError(
                f"Feature '{self.name}' has different specs from source: "
                f"expected {self.spec}, got {source.spec}."
            )

        if not source.owner:
            raise RuntimeError(
                f"Cannot share source feature '{source.name}' data, "
                "it is not an owner of its data."
            )

        if self.owner:
            raise RuntimeError(
                f"Cannot gather data from other feature, feature '{self.name}' "
                "is an owner."
            )

        if is_unset(source._data):
            raise RuntimeError(
                f"Source feature '{source.name}' has no loaded data to share."
            )

        self._data = source._data

    def validate(self, **symbol_values: int) -> None:
        if not self.loaded:
            if not self.optional:
                raise RuntimeError(
                    f"Feature {self.name} is required but has no loaded data."
                )

            return

        target_shape = self.spec.resolve_shape(**symbol_values)
        if target_shape is not None and not has_shape(self._data, target_shape):
            raise ValueError(
                f"Feature {self.name} has invalid shape: "
                f"expected {target_shape}, got {self._data}."
            )

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Feature)
            and self.name == value.name
            and self.spec == value.spec
        )


def expand_shape(
    first_dim: BaseShapeDim, shape: tuple[BaseShapeDim, ...] | None
) -> tuple[BaseShapeDim, ...]:
    if shape is None:
        return (first_dim, None)

    return (first_dim, *shape)


class TaskFeature(Feature[list[_T]]):
    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        *,
        optional: bool = False,
        default: list[_T] | _UnsetType = UNSET,
        owner: bool | None = None,
        dynamic: bool = False,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        scope: Literal["task"] = "task"

        super().__init__(
            name=name,
            scope=scope,
            semantic=semantic,
            optional=optional,
            default=default,
            owner=owner,
            dynamic=dynamic,
            shape=expand_shape("n_tasks", shape),
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
        owner: bool | None = None,
        default: list[_T] | _UnsetType = UNSET,
        dynamic: bool = False,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        scope: Literal["job"] = "job"

        super().__init__(
            name=name,
            scope=scope,
            semantic=semantic,
            optional=optional,
            owner=owner,
            default=default,
            dynamic=dynamic,
            shape=expand_shape("n_jobs", shape),
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
        owner: bool | None = None,
        default: list[_T] | _UnsetType = UNSET,
        dynamic: bool = False,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        scope: Literal["machine"] = "machine"

        super().__init__(
            name=name,
            scope=scope,
            semantic=semantic,
            optional=optional,
            owner=owner,
            default=default,
            dynamic=dynamic,
            shape=expand_shape("n_machines", shape),
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
        owner: bool | None = None,
        default: _T | _UnsetType = UNSET,
        dynamic: bool = False,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        scope: Literal["global"] = "global"

        super().__init__(
            name=name,
            scope=scope,
            semantic=semantic,
            optional=optional,
            owner=owner,
            default=default,
            dynamic=dynamic,
            shape=shape,
            n_categories=n_categories,
        )

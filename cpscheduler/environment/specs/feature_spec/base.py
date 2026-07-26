"""Feature specifications for observations in the scheduling environment."""

from typing import Any, Generic, TypeVar, cast, override

from mypy_extensions import mypyc_attr
from typing_extensions import TypeIs

from cpscheduler.environment.instance.metadata import FeatureMetadata, ValueType
from cpscheduler.environment.specs.base import ObservationSpec
from cpscheduler.environment.utils.symbols import (
    BaseShapeDim,
    SymbolicDim,
    resolve_shape,
    symbolic_shape,
    to_raw_shape,
)

_T = TypeVar("_T")
_R = TypeVar("_R")


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class FeatureViewSpec(ObservationSpec, Generic[_T, _R]):
    """Specification for a view of a feature in the observation."""

    value_type: ValueType
    _shape: tuple[SymbolicDim | None, ...] | None

    n_categories: int | None
    low: float | None
    high: float | None

    def __init__(
        self,
        value_type: ValueType,
        shape: tuple[BaseShapeDim, ...] | None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize the FeatureViewSpec with the given metadata."""
        self.value_type = value_type
        self._shape = symbolic_shape(shape)
        self.n_categories = n_categories
        self.low = low
        self.high = high

    def __repr__(self) -> str:
        """Return a string representation of the feature metadata."""
        attrs = [
            f"value_type={self.value_type!r}",
            f"shape={self.shape!r}",
        ]

        if self.n_categories is not None:
            attrs.append(f"n_categories={self.n_categories!r}")

        if self.low is not None:
            attrs.append(f"low={self.low!r}")

        if self.high is not None:
            attrs.append(f"high={self.high!r}")

        return f"{type(self).__name__}({', '.join(attrs)})"

    @property
    def shape(self) -> tuple[SymbolicDim | None, ...] | None:
        """Get the symbolic shape of the feature view."""
        return self._shape

    @property
    def raw_shape(self) -> tuple[BaseShapeDim, ...] | None:
        """Get the raw shape of the feature view."""
        return to_raw_shape(self._shape)

    def resolve_shape(self, **symbols: int) -> tuple[int | None, ...] | None:
        """Resolve the symbolic shape to a concrete shape using the provided symbols."""
        return resolve_shape(self._shape, **symbols)

    def materialize(self, data: _T, **symbols: int) -> _R:
        """Materialize the feature data using the materialization function."""
        raise NotImplementedError(
            "Materialization function is not implemented for this feature view "
            f"{type(self).__name__}. Please implement the `materialize` method."
        )


# Basic feature view specifications for common feature types
# These are inferrable from the feature metadata alone and do not require
# additional information to materialize the feature data.


def is_dense_shape(
    shape: tuple[BaseShapeDim, ...] | None,
) -> TypeIs[tuple[int | str, ...]]:
    """Check if the shape is fully specified (i.e., no None dimensions)."""
    return shape is not None and all(dim is not None for dim in shape)


class DenseViewSpec(FeatureViewSpec[_T, _T]):
    """A dense view of a feature, where the data is represented as a dense array."""

    def __init__(
        self,
        value_type: ValueType,
        shape: tuple[int | str, ...],
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize the DenseViewSpec with the given metadata."""
        super().__init__(value_type, shape, n_categories, low, high)

        if not is_dense_shape(shape):
            raise ValueError(
                "DenseViewSpec requires a fully specified shape. "
                f"Received shape: {self.shape}"
            )

    @property
    @override
    def shape(self) -> tuple[SymbolicDim, ...]:
        return cast("tuple[SymbolicDim, ...]", self._shape)

    @property
    @override
    def raw_shape(self) -> tuple[int | str, ...]:
        return cast("tuple[int | str, ...]", super().raw_shape)

    def resolve_shape(self, **symbols: int) -> tuple[int, ...]:
        """Resolve the symbolic shape to a concrete shape using the provided symbols."""
        return cast("tuple[int, ...]", super().resolve_shape(**symbols))

    def materialize(self, data: _T, **symbols: int) -> _T:
        """Materialize the feature data as a dense array."""
        return data


class FreeViewSpec(FeatureViewSpec[_T, _T]):
    """A free view of a feature, where the data has no shape constraints."""

    def __init__(
        self,
        value_type: ValueType,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize the FreeViewSpec with the given metadata."""
        super().__init__(
            value_type,
            shape=None,
            n_categories=n_categories,
            low=low,
            high=high,
        )

    @property
    @override
    def shape(self) -> None:
        return None

    @property
    @override
    def raw_shape(self) -> None:
        return None

    def resolve_shape(self, **symbols: int) -> None:
        """Resolve the symbolic shape to a concrete shape using the provided symbols."""

    def materialize(self, data: _T, **symbols: int) -> _T:
        """Materialize the feature data without any shape constraints."""
        return data


class RaggedViewSpec(FeatureViewSpec[_T, _T]):
    """A ragged view of a feature, where the data have variable-length dimensions."""

    def __init__(
        self,
        value_type: ValueType,
        shape: tuple[BaseShapeDim, ...],
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize the RaggedViewSpec with the given metadata."""
        super().__init__(value_type, shape, n_categories, low, high)

        if is_dense_shape(shape):
            raise ValueError(
                "RaggedViewSpec requires at least one variable-length dimension. "
                f"Received shape: {self.shape}"
            )

    @property
    @override
    def shape(self) -> tuple[SymbolicDim | None, ...]:
        return cast("tuple[SymbolicDim | None, ...]", self._shape)

    @property
    @override
    def raw_shape(self) -> tuple[BaseShapeDim, ...]:
        return cast("tuple[BaseShapeDim, ...]", super().raw_shape)

    def resolve_shape(self, **symbols: int) -> tuple[int | None, ...]:
        """Resolve the symbolic shape to a concrete shape using the provided symbols."""
        return cast("tuple[int | None, ...]", super().resolve_shape(**symbols))

    def materialize(self, data: _T, **symbols: int) -> _T:
        """Materialize the feature data as a ragged array."""
        return data


def from_metadata(metadata: FeatureMetadata) -> FeatureViewSpec[Any, Any]:
    """Create a FeatureViewSpec from the given FeatureMetadata."""
    shape = metadata.raw_shape

    if is_dense_shape(shape):
        return DenseViewSpec(
            value_type=metadata.value_type,
            shape=shape,
            n_categories=metadata.n_categories,
            low=metadata.low,
            high=metadata.high,
        )

    if shape is None:
        return FreeViewSpec(
            value_type=metadata.value_type,
            n_categories=metadata.n_categories,
            low=metadata.low,
            high=metadata.high,
        )

    return RaggedViewSpec(
        value_type=metadata.value_type,
        shape=shape,
        n_categories=metadata.n_categories,
        low=metadata.low,
        high=metadata.high,
    )

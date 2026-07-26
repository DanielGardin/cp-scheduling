"""Feature metadata class to store information about the feature data."""

from typing import Any, Literal

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.utils.symbols import (
    BaseShapeDim,
    SymbolicDim,
    resolve_shape,
    solve_shape,
    symbolic_shape,
    to_raw_shape,
)

ValueType = Literal[
    # Numeric
    "continuous",  # Real-valued, unbounded
    "discrete",  # Integer-valued, unbounded
    "binary",  # Boolean-valued, {0, 1}
    "count",  # Non-negative integer-valued
    # Bounded numeric
    "normalized",  # Real-valued, bounded in [low, high]
    "probability",  # Real-valued, bounded in [0, 1], sum to 1
    # Scheduling quantities
    "time",  # Integer-valued, bounded in [MIN_TIME, MAX_TIME]
    "duration",  # Non-negative integer-valued time duration
    "cost",  # Non-negative real-valued cost
    # Identifiers
    "id",  # Integer-valued, unique identifier
    "task_id",  # Integer-valued, bounded in [0, n_tasks)
    "job_id",  # Integer-valued, bounded in [0, n_jobs)
    "machine_id",  # Integer-valued, bounded in [0, n_machines)
    # Ordered / categorical
    "order",  # Non-negative integer-valued, teorically bounded
    "categorical",  # Categorical, integer-valued, bounded in [0, n_categories)
    # Unknown
    "unknown",  # Non-structured, or non-scalar
]

_BOUNDS: dict[ValueType, tuple[float, float | None]] = {
    "probability": (0.0, 1.0),
    "binary": (0.0, 1.0),
    "normalized": (0.0, 1.0),
    "count": (0.0, None),
    "duration": (0.0, None),
    "cost": (0.0, None),
    "time": (0.0, None),
    "order": (0.0, None),
    "task_id": (0.0, None),
    "job_id": (0.0, None),
    "machine_id": (0.0, None),
}


class FeatureMetadata(EzPickle):
    """Metadata for a scheduling instance feature."""

    value_type: ValueType
    shape: tuple[SymbolicDim | None, ...] | None

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
        """Initialize feature metadata.

        Parameters
        ----------
        value_type: ValueType
            The type of values the feature holds, used for validation and interpretation.

        shape: tuple[BaseShapeDim, ...] or None
            The shape of the feature data, where BaseShapeDim can be an int or a
            symbolic dimension. If None, the shape is not specified.

        n_categories: int | None, default None
            The number of categories, if the feature is categorical.

        low: float | None, default None
            The lower bound of the feature values, if applicable.

        high: float | None, default None
            The upper bound of the feature values, if applicable.

        """
        self.value_type = value_type
        self.shape = symbolic_shape(shape)
        self.n_categories = n_categories

        default_low, default_high = self._infer_bounds()

        self.low = low if low is not None else default_low
        self.high = high if high is not None else default_high

    def _infer_bounds(self) -> tuple[float | None, float | None]:
        """Infer the bounds of the feature based on its value type."""
        if self.value_type == "categorical":
            if self.n_categories is None:
                raise ValueError(
                    "'n_categories' must be provided for categorical features."
                )
            return 0.0, float(self.n_categories - 1)

        return _BOUNDS.get(self.value_type, (None, None))

    def __repr__(self) -> str:
        """Return a string representation of the feature metadata."""
        attrs = [
            f"value_type={self.value_type!r}",
            f"shape={self.shape!r}",
        ]

        if self.n_categories is not None:
            attrs.append(f"n_categories={self.n_categories!r}")

        else:
            if self.low is not None:
                attrs.append(f"low={self.low!r}")

            if self.high is not None:
                attrs.append(f"high={self.high!r}")

        return f"{type(self).__name__}({', '.join(attrs)})"

    @property
    def symbols(self) -> set[str]:
        """Return the set of symbols used in the feature's shape."""
        if self.shape is None:
            return set()

        symbols: set[str] = set()
        for dim in self.shape:
            if isinstance(dim, SymbolicDim):
                symbols.update(dim.symbols)

        return symbols

    @property
    def raw_shape(self) -> tuple[BaseShapeDim, ...] | None:
        """Return the unresolved shape."""
        return to_raw_shape(self.shape)

    def resolve_shape(self, **symbols: int) -> tuple[int | None, ...] | None:
        """Resolve the symbolic dimensions in the feature's shape to concrete integers."""
        return resolve_shape(self.shape, **symbols)

    def solve_symbols(self, data: Any) -> dict[str, int]:
        """Solve the shape of the feature's data, inferring symbolic dimensions if necessary."""
        shape = self.shape
        if shape is None:
            return {}

        return solve_shape(shape, data) or {}

    def is_compatible(self, other: "FeatureMetadata") -> bool:
        """Check if this feature metadata is compatible with another.

        Two features are considered compatible if they have the same value type
        and their shapes can be resolved to the same concrete shape.

        Parameters
        ----------
        other: FeatureMetadata
            The other feature metadata to compare against.

        Returns
        -------
        bool
            True if the features are compatible, False otherwise.

        """
        return (
            self.value_type == other.value_type
            and self.raw_shape == other.raw_shape
        )

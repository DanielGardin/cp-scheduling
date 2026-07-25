"""Feature specifications for observations in the scheduling environment."""

from collections.abc import Callable
from typing import Generic, Literal, TypeVar, cast

from cpscheduler.environment.specs.base import ObservationSpec
from cpscheduler.environment.utils.symbols import (
    BaseShapeDim,
    SymbolicDim,
    resolve_shape,
    symbolic_shape,
)

RepresentationKind = Literal[
    # Flat tensor-like representations
    "dense",  # Default Complete shape with len(shape) > 0
    "scalar",  # Default Single value with shape == ()
    # Variable-length representations
    "sequence",  # Default Any shape with a None dimension
    # Sparse representations
    "edge_list",
    "coordinate_list",  # COO-like
    "compressed",  # CSR/CSC/etc.
    # Binary indicator representations
    "mask",  # Default Binary mask with Complete shape and binary values
    "set",
    # Matrix representations
    "adjacency",
    "incidence",
    # Interval-based
    "interval",
    "calendar",
    # Generic graph
    "graph",
    # Unknown / opaque
    "opaque",  # Default Non-structured, with None shape
]

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


def _infer_representation(
    value_type: ValueType, shape: tuple[BaseShapeDim, ...] | None
) -> RepresentationKind:
    """Infer the representation kind based on value type and shape."""
    if shape is None:
        return "opaque"

    if len(shape) == 0:
        return "scalar"

    if any(dim is None for dim in shape):
        return "sequence"

    if value_type == "binary":
        return "mask"

    return "dense"


_T = TypeVar("_T")
_R = TypeVar("_R")


class FeatureViewSpec(ObservationSpec, Generic[_T, _R]):
    """Specification for a view of a feature in the observation.

    A FeatureViewSpec describes how a feature is represented in the observation space.
    It includes the scope, semantic type, and shape of the feature.

    """

    representation: RepresentationKind
    value_type: ValueType
    shape: tuple[SymbolicDim | None, ...] | None
    _materialize_fn: Callable[[_T], _R] | None = None

    # View metadata
    n_categories: int | None
    low: float | None
    high: float | None

    def __init__(
        self,
        value_type: ValueType,
        representation: RepresentationKind | None = None,
        *,
        materialize_fn: Callable[[_T], _R] | None = None,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize a feature view specification.

        Parameters
        ----------
        value_type: ValueType
            The semantic type of the feature (e.g., "continuous", "categorical").

        representation: RepresentationKind | None, default None
            The representation kind of the feature (e.g., "dense", "sequence").
            If None, a default representation will be inferred based on the
            value_type and shape.

        materialize_fn: Callable[[_T], _R] | None, default None
            A function to materialize the feature data from its raw form (_T)
            to its processed form (_R). If None, the materialization will be
            assumed to be the identity function.

        shape: tuple[BaseShapeDim, ...] | None, default None
            The shape of the feature, if it is an array.
            Each dimension can be an integer, a symbolic dimension (e.g. "n_tasks"),
            or None for non-structured dimensions.

        n_categories: int | None, default None
            The number of categories, if the feature is categorical.

        low: float | None, default None
            The lower bound of the feature values, if applicable.

        high: float | None, default None
            The upper bound of the feature values, if applicable.

        Note
        ----
        `value_type` is used to infer the values of `low` and `high` if they
        are not provided.
        For example, if `value_type` is "normalized", then `low` will default
        to 0.0 and `high` will default to 1.0.
        Whereas, if `value_type` is "continuous", then `low` and `high` will default
        to None, indicating unbounded values.

        Raises
        ------
        ValueError
            If n_categories is provided for a non-categorical feature.
            If low and high are both provided and high < low.

        """
        if low is not None and high is not None and high < low:
            raise ValueError(f"Expected low <= high, got {low} > {high}.")

        self.value_type = value_type
        self.representation = representation or _infer_representation(
            value_type, shape
        )
        self._materialize_fn = materialize_fn

        self.shape = symbolic_shape(shape)

        self.n_categories = n_categories

        inferred_low, inferred_high = self._infer_bounds()

        self.low = inferred_low or low
        self.high = inferred_high or high

    def _infer_bounds(self) -> tuple[float | None, float | None]:
        """Infer the bounds of the feature based on its value type."""
        match self.value_type:
            case "probability" | "binary":
                return 0.0, 1.0

            case "normalized":
                return None, None  # Bounds are provided by the user

            case "count" | "duration" | "cost" | "time" | "order":
                return 0.0, None

            case "categorical":
                if self.n_categories is None:
                    raise ValueError(
                        "'n_categories' must be provided for categorical views."
                    )

                return 0.0, float(self.n_categories - 1)

            case "task_id" | "job_id" | "machine_id":
                return 0.0, None

            case _:
                return None, None

    @property
    def symbols(self) -> set[str]:
        """Return the symbolic dimensions used by this view."""
        symbols: set[str] = set()

        if self.shape is not None:
            for dim in self.shape:
                if dim is not None:
                    symbols.update(dim.symbols)

        return symbols

    @property
    def raw_shape(self) -> tuple[BaseShapeDim, ...] | None:
        """Return the unresolved shape."""
        if self.shape is None:
            return None

        return tuple(
            dim.raw if isinstance(dim, SymbolicDim) else None
            for dim in self.shape
        )

    def resolve_shape(
        self, **symbol_values: int
    ) -> tuple[int | None, ...] | None:
        """Resolve symbolic dimensions."""
        return resolve_shape(self.shape, **symbol_values)

    def materialize(self, data: _T) -> _R:
        """Materialize the feature data using the provided materialization function."""
        if self._materialize_fn is None:
            return cast("_R", data)

        return self._materialize_fn(data)

    def __repr__(self) -> str:
        """Return a string representation of the FeatureViewSpec."""
        attrs = [
            f"representation={self.representation!r}",
            f"value_type={self.value_type!r}",
        ]

        if self.shape is not None:
            attrs.append(f"shape={self.shape!r}")

        if self.n_categories is not None:
            attrs.append(f"n_categories={self.n_categories!r}")

        if self.low is not None:
            attrs.append(f"low={self.low!r}")

        if self.high is not None:
            attrs.append(f"high={self.high!r}")

        return f"FeatureViewSpec({', '.join(attrs)})"

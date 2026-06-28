"""Feature specifications for observations in the scheduling environment."""

from typing import Literal

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.specs.symbols import (
    BaseShapeDim,
    SymbolicDim,
    resolve_shape,
    symbolic_shape,
)


class ObservationSpec(EzPickle):
    """Base specification node for observation structures."""


SemanticType = Literal[
    # Numerical
    "continuous",  # Real number
    "discrete",  # Integer number
    "binary",  # Boolean
    "count",  # Non-negative integers
    "cost",  # Non-negative real
    "probability",  # Real [0,1], interpreted as p(.)
    "normalized",  # Real [0,1]
    # Identity / indexing
    "id",  # ID Integer
    "order",  # Ordinal Integer
    "categorical",  # Arbitrary categories
    # Scheduling
    "task",  # Integer [0, n_tasks)
    "job",  # Integer [0, n_jobs)
    "machine",  # Integer [0, n_machines)
    "time",  # Integer [0, MAX_TIME)
    "duration",  # Integer delta [0, MAX_TIME)
    "interval",  # Interval (start, end)
    "calendar",  # Sequence of intervals (start, end)
    # Structural
    "mask",  # Boolean mask
    "set",  # Arbitrary set. Represented as a binary indicator vector of possible elements.
    # Graph
    "adjacency",  # Adjacency matrix
    # Unknown
    "unknown",  # Non-structured data, ignored when serialized
]

Scope = Literal[
    "task",
    "job",
    "machine",
    "global",
]


class FeatureSpec(ObservationSpec):
    """Specification for a single feature in the observation."""

    scope: Scope
    semantic: SemanticType
    sparse: bool

    # Feature metadata (used for ObservationSpec)
    shape: tuple[SymbolicDim | None, ...] | None
    n_categories: int | None
    low: float | None
    high: float | None

    def __init__(
        self,
        scope: Scope,
        semantic: SemanticType,
        sparse: bool = False,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize a feature specification.

        Parameters
        ----------
        scope: {"task", "job", "machine", "global"}
            The scope of the feature (task, job, machine, global).

        semantic: SemanticType
            The semantic type of the feature, which determines how it should be
            interpreted and processed.

        sparse: bool, default False
            Whether the feature is sparse (i.e. mostly zeros) and should be stored
            in a sparse format.

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

        Raises
        ------
        ValueError
            If n_categories is provided for a non-categorical feature.
            If low and high are both provided and high < low.

        """
        if n_categories is not None and semantic != "categorical":
            raise ValueError(
                f"Cannot provide 'n_categories' in a '{semantic}' feature."
            )

        if low is not None and high is not None and high < low:
            raise ValueError(f"Expected low < high, but {low} > {high}.")

        self.scope = scope
        self.semantic = semantic
        self.sparse = sparse

        self.shape = symbolic_shape(shape)

        self.n_categories = n_categories
        self.low = low
        self.high = high

    @property
    def raw_shape(self) -> tuple[BaseShapeDim, ...] | None:
        """Return the raw shape of the feature, defined by strings and integers."""
        if self.shape is None:
            return None

        return tuple(
            dim.raw if isinstance(dim, SymbolicDim) else None
            for dim in self.shape
        )

    @property
    def symbols(self) -> set[str]:
        """Return the symbolic dimensions of the shape."""
        symbols: set[str] = set()

        if self.shape:
            for dim in self.shape:
                if dim:
                    symbols.update(dim.symbols)

        return symbols

    # FUTURE: Implement broadcasted views for shareable features
    # This method is also incomplete, shapes are not compared, as it is not
    # obvious how the broadcast must happen.
    # For example, should a (n_jobs, n_jobs) be sharable only with (n_tasks, n_jobs),
    # or with (n_tasks, n_tasks) as well?
    def shareable_with(self, other: "FeatureSpec") -> bool:
        """Check if this feature spec can share the same underlying data as another spec.

        To be shareable, both specs must have the same semantic and metadata
        ( n_categories, low, high).
        Additionally, the scopes must be compatible, i.e. one must be a subset
        of the other.

        The possible scope combinations are:
        - S <-> S, for any scope S
        - job -> task (job-level features can be shared with task-level features, but not vice versa)
        - global -> S, for any scope S (global features can be shared with any scope, but not vice versa)
        """
        return (
            self.semantic == other.semantic
            and self.n_categories == other.n_categories
            and self.low == other.low
            and self.high == other.high
            and (
                self.scope == other.scope
                or (self.scope == "job" and other.scope == "task")
                or self.scope == "global"
            )
        )

    def resolve_shape(
        self, **symbol_values: int
    ) -> tuple[int | None, ...] | None:
        """Resolve the symbolic dimensions in the shape to concrete integers using the provided symbol values."""
        return resolve_shape(self.shape, **symbol_values)

    def __eq__(self, value: object, /) -> bool:
        """Check equality of FeatureSpecs."""
        return (
            isinstance(value, FeatureSpec)
            and self.scope == value.scope
            and self.semantic == value.semantic
            and self.sparse == value.sparse
            and self.shape == value.shape
            and self.n_categories == value.n_categories
            and self.low == value.low
            and self.high == value.high
        )

    def __hash__(self) -> int:
        """Hash based on the attributes of the FeatureSpec."""
        return hash(
            (
                self.scope,
                self.semantic,
                self.sparse,
                self.shape,
                self.n_categories,
                self.low,
                self.high,
            )
        )

from typing import Literal

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.specs.symbols import BaseShapeDim, SymbolicDim


class ObservationSpec(EzPickle):
    """
    Base specification node for observation structures.
    """


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
    scope: Scope
    semantic: SemanticType
    sparse: bool
    optional: bool

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
        optional: bool = False,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        if n_categories is not None and semantic != "categorical":
            raise ValueError(
                f"Cannot provide 'n_categories' in a '{semantic}' feature."
            )

        if low is not None and high is not None and high < low:
            raise ValueError(f"Expected low < high, but {low} > {high}.")

        self.scope = scope
        self.semantic = semantic
        self.sparse = sparse
        self.optional = optional

        self.shape = None
        if shape is not None:
            self.shape = tuple(
                SymbolicDim.from_shapedim(dim) if dim is not None else dim
                for dim in shape
            )

        self.n_categories = n_categories
        self.low = low
        self.high = high

    @property
    def raw_shape(self) -> tuple[BaseShapeDim, ...] | None:
        if self.shape is None:
            return None

        return tuple(
            dim.raw if isinstance(dim, SymbolicDim) else dim
            for dim in self.shape
        )

    def shareable_with(self, other: "FeatureSpec") -> bool:
        """Check if this feature spec can share the same underlying data as
        another spec.

        To be shareable, both specs must have the same semantic and metadata
        (shape, n_categories, low, high).
        Additionally, the scopes must be compatible, i.e. one must be a subset
        of the other.
        The possible scope combinations are:
        - S <-> S, for any scope S
        - job -> task (job-level features can be shared with task-level features, but not vice versa)
        - global -> S, for any scope S (global features can be shared with any scope, but not vice versa)
        """
        return (
            self.semantic == other.semantic
            and self.shape == other.shape
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
        if self.shape is None:
            return None

        return tuple(
            dim.resolve(**symbol_values)
            if isinstance(dim, SymbolicDim)
            else dim
            for dim in self.shape
        )

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, FeatureSpec)
            and self.scope == value.scope
            and self.semantic == value.semantic
            and self.sparse == value.sparse
            and self.optional == value.optional
            and self.shape == value.shape
            and self.n_categories == value.n_categories
            and self.low == value.low
            and self.high == value.high
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.scope,
                self.semantic,
                self.sparse,
                self.optional,
                self.shape,
                self.n_categories,
                self.low,
                self.high,
            )
        )

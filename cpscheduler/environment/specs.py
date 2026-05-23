# from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal

from cpscheduler.environment.constants import EzPickle

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
    "set",  # Arbitrary set
    "sequence",  # Arbitrary sequence
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

ShapeDim = (
    int
    | Literal[
        "n_tasks",
        "n_jobs",
        "n_machines",
    ]
)

DType = type[int] | type[float] | type[bool] | type[str]

BOUNDS: dict[SemanticType, tuple[float | None, float | None]] = {
    "continuous": (None, None),
    "discrete": (None, None),
    "binary": (0, 1),
    "count": (0, None),
    "cost": (0, None),
    "probability": (0, 1),
    "normalized": (0, 1),
    "id": (0, None),
    "order": (0, None),
    "categorical": (None, None),
    "task": (0, None),
    "job": (0, None),
    "machine": (0, None),
    "time": (0, None),
    "duration": (0, None),
    "interval": (0, None),  # TODO: separate start/end bounds
    "calendar": (0, None),  # TODO: separate start/end bounds
    "mask": (0, 1),
    "set": (None, None),
    "sequence": (None, None),
    "adjacency": (0, 1),
    "unknown": (None, None),
}


class ObservationSpec(EzPickle):
    """
    Base specification node for observation structures.
    """


# FUTURE: Support sparse features, i.e. dict[SupportIndex, T]
# Materialization then materialize a list[T].
class FeatureSpec(ObservationSpec):
    scope: Scope
    semantic: SemanticType
    sparse: bool
    optional: bool

    # Feature metadata (used for ObservationSpec)
    dtype: DType | None
    shape: tuple[ShapeDim, ...]
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
        dtype: DType | None = None,
        shape: tuple[ShapeDim, ...] = (),
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

        self.dtype = dtype
        self.shape = shape
        self.n_categories = n_categories
        self.low = low
        self.high = high

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, FeatureSpec)
            and self.scope == value.scope
            and self.semantic == value.semantic
            and self.sparse == value.sparse
            and self.optional == value.optional
            and self.dtype == value.dtype
            and self.shape == value.shape
            and self.n_categories == value.n_categories
            and self.low == value.low
            and self.high == value.high
        )


class DictSpec(ObservationSpec):
    fields: dict[str, ObservationSpec]

    def __init__(self, fields: Mapping[str, ObservationSpec]) -> None:
        self.fields = dict(fields)


class SequenceSpec(ObservationSpec):
    element: ObservationSpec
    length: int | None

    def __init__(
        self, element: ObservationSpec, length: int | None = None
    ) -> None:
        self.element = element
        self.length = length


class GraphSpec(ObservationSpec):
    nodes: ObservationSpec
    edges: ObservationSpec | None
    graph: FeatureSpec

    def __init__(
        self,
        nodes: ObservationSpec,
        edges: ObservationSpec | None,
        graph: FeatureSpec,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.graph = graph

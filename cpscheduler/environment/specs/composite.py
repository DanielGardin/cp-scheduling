"""Composite observation specifications for complex observations.

This module defines composite observation specifications that combine multiple
features or observations into structured formats, such as stacks, dictionaries,
sequences, and graphs.

These composite specs allow for representing complex observations that can be
easily serialized, statically analyzed and validated.
"""

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Literal

from cpscheduler.environment.specs.feature_spec import (
    FeatureSpec,
    ObservationSpec,
)
from cpscheduler.environment.specs.symbols import (
    ShapeDim,
    SymbolicDim,
    resolve_shape,
)


def _remove_dim(
    shape: tuple[SymbolicDim | None, ...],
    feature_dim: int,
) -> tuple[SymbolicDim | None, ...]:
    return shape[:feature_dim] + shape[feature_dim + 1 :]


class StackSpec(ObservationSpec):
    """Observation spec for stacking multiple features along a new dimension.

    Often used for representing feature matrices, where multiple features are
    stacked along the feature dimension.
    """

    features: tuple[FeatureSpec, ...]
    shape: tuple[SymbolicDim | None, ...]

    def __init__(
        self,
        features: list[FeatureSpec],
        feature_dim: int = -1,
    ) -> None:
        """Initialize a StackSpec.

        Parameters
        ----------
        features: list[FeatureSpec]
            The list of feature specifications to stack. All features must have
            the same shape except for the dimension along which they are stacked.

        feature_dim: int, default -1
            The dimension along which to stack the features. Can be negative to
            index from the end (e.g. -1 for the last dimension).

        Raises
        ------
        ValueError
            If the features list is empty, or if the features have incompatible shapes.

        IndexError
            If the feature_dim is out of bounds for the feature shapes.

        """
        if not features:
            raise ValueError("StackSpec requires at least one feature.")

        self.features = tuple(features)

        stack_shape = features[0].shape

        if stack_shape is None:
            raise ValueError(
                f"Cannot stack shapeless features. Feature '{features[0]}'."
            )

        rank = len(stack_shape)

        if feature_dim < 0:
            feature_dim += rank

        if not 0 <= feature_dim < rank:
            raise IndexError(
                f"Invalid feature dimension {feature_dim} for rank {rank}."
            )

        base_shape = _remove_dim(stack_shape, feature_dim)
        stacked_dim = SymbolicDim()

        for i, feature in enumerate(features[1:], start=1):
            shape = feature.shape

            if shape is None:
                raise ValueError(
                    f"Cannot stack shapeless features. Feature '{features[0]}'."
                )

            if _remove_dim(shape, feature_dim) != base_shape:
                raise ValueError(
                    "All features in a StackSpec must "
                    "match except for the stacked dimension. "
                    f"Expected {stack_shape}, "
                    f"got {feature.shape} "
                    f"for feature {i}."
                )

            stacked_dim += shape[feature_dim]

        self.shape = (
            *stack_shape[:feature_dim],
            stacked_dim,
            *stack_shape[feature_dim + 1 :],
        )

    def resolve_shape(self, **symbol_values: int) -> tuple[int | None, ...]:
        """Resolve the symbolic dimensions in the shape to concrete integers using the provided symbol values."""
        return resolve_shape(self.shape, **symbol_values)

    def __eq__(self, value: object, /) -> bool:
        """Check equality of StackSpecs."""
        return isinstance(value, StackSpec) and self.features == value.features

    def __hash__(self) -> int:
        """Hash based on the attributes of the StackSpecs."""
        return hash(self.features)


class DictSpec(ObservationSpec, Mapping[str, ObservationSpec]):
    """Observation spec for a dictionary of features, where each feature is accessed by a string key.

    Often used for representing structured observations, where different features
    are accessed by name.
    """

    _fields: dict[str, ObservationSpec]

    def __init__(self, fields: Mapping[str, ObservationSpec]) -> None:
        """Initialize a DictSpec.

        Parameters
        ----------
        fields: Mapping[str, ObservationSpec]
            A mapping from string keys to observation specifications for each field.

        """
        self._fields = dict(fields)

    def __getitem__(self, key: str) -> ObservationSpec:
        """Get the observation spec for a given field key."""
        return self._fields[key]

    def items(self) -> ItemsView[str, ObservationSpec]:
        """Return a view of the field items (key-spec pairs)."""
        return self._fields.items()

    def keys(self) -> KeysView[str]:
        """Return a view of the field keys."""
        return self._fields.keys()

    def values(self) -> ValuesView[ObservationSpec]:
        """Return a view of the field observation specs."""
        return self._fields.values()

    def __iter__(self) -> Iterator[str]:
        """Iterate over the field keys."""
        return iter(self._fields)

    def __len__(self) -> int:
        """Return the number of fields in the DictSpec."""
        return len(self._fields)

    def __eq__(self, value: object, /) -> bool:
        """Check equality of DictSpecs."""
        return isinstance(value, DictSpec) and self._fields == value._fields

    def __hash__(self) -> int:
        """Hash based on the attributes of the DictSpecs."""
        return hash(frozenset(self._fields.items()))


class SequenceSpec(ObservationSpec):
    """Observation spec for a sequence of features.

    Often used for representing variable-length sequences of features, such as
    the sequence of tasks in a job, or the sequence of jobs in a schedule.

    Note that this observation overlaps with FeatureSpec, since a feature with
    a variable-dimension value is supported by both.
    This spec is meant for cases where the element is not just a feature, but a
    more complex observation, such as a DictSpec or GraphSpec, that is repeated
    in a sequence.

    If the element is a simple feature, it may be more efficient to stack them
    using a StackSpec, or changing the feature shape to include the sequence
    dimension, rather than using a SequenceSpec.
    """

    element: ObservationSpec
    length: SymbolicDim | None

    def __init__(
        self, element: ObservationSpec, length: "ShapeDim | None" = None
    ) -> None:
        """Initialize a SequenceSpec.

        Parameters
        ----------
        element: ObservationSpec
            The observation specification for each element in the sequence.

        length: ShapeDim | None, default None
            An optional symbolic dimension representing the length of the sequence.
            If None, the sequence is assumed to have unbounded length.

        """
        self.element = element
        self.length = (
            (
                SymbolicDim.from_shapedim(length)
                if isinstance(length, int | str)
                else length
            )
            if length is not None
            else None
        )

    def __eq__(self, value: object, /) -> bool:
        """Check equality of SequenceSpecs."""
        return (
            isinstance(value, SequenceSpec)
            and self.element == value.element
            and self.length == value.length
        )

    def __hash__(self) -> int:
        """Hash based on the attributes of the SequenceSpec."""
        return hash((self.element, self.length))


GraphRepresentation = Literal[
    "adjacency_list",
    "adjacency_matrix",
    "incidence_matrix",
    "edge_list",
]


class GraphSpec(ObservationSpec):
    """Observation spec for a graph-structured observation.

    Commonly used for representing scheduling problems as graphs, where nodes
    represent tasks or jobs, and edges represent ordering constraints or conflicts
    between tasks.
    """

    nodes: StackSpec
    edges: StackSpec
    representation: GraphRepresentation

    def __init__(
        self,
        nodes: StackSpec,
        edges: StackSpec,
        representation: GraphRepresentation = "adjacency_list",
    ) -> None:
        """Initialize a GraphSpec.

        Parameters
        ----------
        nodes: StackSpec
            A StackSpec representing the features of the graph nodes. The first
            dimension of the node features should correspond to the number of nodes.

        edges: StackSpec
            A StackSpec representing the features of the graph edges. The first
            dimension of the edge features should correspond to the number of edges.

        representation: GraphRepresentation, default "adjacency_list"
            The representation format of the graph. This can be used to indicate
            how the nodes and edges should be interpreted (e.g. as an adjacency list,
            adjacency matrix, etc.).

            This is not strictly necessary for the GraphSpec itself, but can be useful for
            validation, and for downstream processing that may need to know the graph format.

        """
        self.nodes = nodes
        self.edges = edges
        self.representation = representation

    def __eq__(self, value: object, /) -> bool:
        """Check equality of GraphSpecs."""
        return (
            isinstance(value, GraphSpec)
            and self.nodes == value.nodes
            and self.edges == value.edges
            and self.representation == value.representation
        )

    def __hash__(self) -> int:
        """Hash based on the attributes of the GraphSpec."""
        return hash((self.nodes, self.edges, self.representation))

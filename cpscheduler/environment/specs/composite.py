from collections.abc import Mapping

from cpscheduler.environment.specs.feature_spec import (
    FeatureSpec,
    ObservationSpec,
)
from cpscheduler.environment.specs.symbols import ShapeDim, SymbolicDim


def _remove_dim(
    shape: tuple[SymbolicDim | None, ...],
    feature_dim: int,
) -> tuple[SymbolicDim | None, ...]:
    return shape[:feature_dim] + shape[feature_dim + 1 :]


class StackSpec(ObservationSpec):
    features: list[FeatureSpec]
    shape: tuple[SymbolicDim | None, ...]

    def __init__(
        self,
        features: list[FeatureSpec],
        feature_dim: int = -1,
    ) -> None:
        if not features:
            raise ValueError("StackSpec requires at least one feature.")

        self.features = features

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

        self.shape = tuple(
            [
                *stack_shape[:feature_dim],
                stacked_dim,
                *stack_shape[feature_dim + 1 :],
            ]
        )

    def resolve_shape(self, **symbol_values: int) -> tuple[int | None, ...]:
        return tuple(
            dim.resolve(**symbol_values)
            if isinstance(dim, SymbolicDim)
            else None
            for dim in self.shape
        )


class DictSpec(ObservationSpec):
    fields: dict[str, ObservationSpec]

    def __init__(self, fields: Mapping[str, ObservationSpec]) -> None:
        self.fields = dict(fields)


class SequenceSpec(ObservationSpec):
    element: ObservationSpec
    length: SymbolicDim | None

    def __init__(
        self, element: ObservationSpec, length: "ShapeDim | None" = None
    ) -> None:
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


class GraphSpec(ObservationSpec):
    nodes: StackSpec
    edges: StackSpec

    def __init__(
        self,
        nodes: StackSpec,
        edges: StackSpec,
    ) -> None:
        self.nodes = nodes
        self.edges = edges

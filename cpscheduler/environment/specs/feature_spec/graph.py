"""Graph feature specifications for observations in the scheduling environment."""

from typing import cast, override

from cpscheduler.environment.instance.metadata import ValueType
from cpscheduler.environment.specs.feature_spec.base import FeatureViewSpec
from cpscheduler.environment.utils.symbols import SymbolicDim

ADJ_LIST = dict[int, list[int]]
ADJ_MATRIX = list[list[bool]]


class AdjacencyMatrixViewSpec(FeatureViewSpec[ADJ_LIST, ADJ_MATRIX]):
    """Specification for a view of an adjacency matrix feature in the observation."""

    def __init__(
        self,
        value_type: ValueType,
        n_nodes: int | str,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize the AdjacencyMatrixViewSpec with the given metadata."""
        super().__init__(
            value_type=value_type,
            shape=(n_nodes, n_nodes),
            n_categories=n_categories,
            low=low,
            high=high,
        )

    @property
    @override
    def shape(self) -> tuple[SymbolicDim, SymbolicDim]:
        return cast("tuple[SymbolicDim, SymbolicDim]", self._shape)

    @property
    @override
    def raw_shape(self) -> tuple[int | str, int | str]:
        return cast("tuple[int | str, int | str]", super().raw_shape)

    @override
    def materialize(
        self, data: dict[int, list[int]], **symbols: int
    ) -> list[list[bool]]:
        n_nodes = self.shape[0].resolve(**symbols)

        matrix = [[False] * n_nodes for _ in range(n_nodes)]

        for node, neighbors in data.items():
            for neighbor in neighbors:
                matrix[node][neighbor] = True

        return matrix

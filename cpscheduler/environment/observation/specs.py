from dataclasses import dataclass

from cpscheduler.environment.instance.features import SemanticType


@dataclass(frozen=True)
class ObservationSpec:
    """
    Base specification node for observation structures.
    """

@dataclass(frozen=True)
class TensorSpec(ObservationSpec):
    shape: tuple[int | None, ...]
    dtype: type
    semantic: SemanticType

@dataclass(frozen=True)
class DictSpec(ObservationSpec):
    fields: dict[str, ObservationSpec]

@dataclass(frozen=True)
class SequenceSpec(ObservationSpec):
    element: ObservationSpec
    length: int | None

@dataclass(frozen=True)
class GraphSpec(ObservationSpec):
    nodes: ObservationSpec
    edges: ObservationSpec | None
    adjacency: ObservationSpec

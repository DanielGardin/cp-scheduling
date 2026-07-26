"""Module for environment specifications.

The observation specifications are used to define the static structure for
observations, which can be used for validation and to inform the agent about
the expected format of the observations.
"""

__all__ = [
    "AdjacencyMatrixViewSpec",
    "DenseViewSpec",
    "DictSpec",
    "FeatureViewSpec",
    "FreeViewSpec",
    "GraphSpec",
    "ObservationSpec",
    "RaggedViewSpec",
    "SequenceSpec",
    "StackSpec",
    "from_metadata",
]


from .base import ObservationSpec
from .composite import DictSpec, GraphSpec, SequenceSpec, StackSpec
from .feature_spec import (
    AdjacencyMatrixViewSpec,
    DenseViewSpec,
    FeatureViewSpec,
    FreeViewSpec,
    RaggedViewSpec,
    from_metadata,
)

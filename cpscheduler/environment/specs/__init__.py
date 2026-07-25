"""Module for environment specifications.

The observation specifications are used to define the static structure for
observations, which can be used for validation and to inform the agent about
the expected format of the observations.
"""

__all__ = [
    "DictSpec",
    "FeatureViewSpec",
    "GraphSpec",
    "ObservationSpec",
    "SequenceSpec",
    "StackSpec",
]


from .base import ObservationSpec
from .composite import DictSpec, GraphSpec, SequenceSpec, StackSpec
from .feature_spec import FeatureViewSpec

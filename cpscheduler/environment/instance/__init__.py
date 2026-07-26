"""Module for problem instance features and the problem instance itself.

Features are used to store information about the problem instance, such as
deadlines, processing times, machine capabilities, etc. They are used by
all components of the environment, and by observations.

The `ProblemInstance` class is the main class for representing a scheduling problem instance.
It contains all the features of the instance, centralizing the information in one place.
"""

__all__ = ["DAGFeature", "Feature", "FeatureMetadata", "ProblemInstance"]

from .features import DAGFeature, Feature
from .instance import ProblemInstance
from .metadata import FeatureMetadata

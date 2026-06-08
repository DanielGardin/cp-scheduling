"""Observation classes for the scheduling environment.

Observations are responsible for extracting relevant information from the environment
state and providing it in a structured format that can be used by learning algorithms
or other components of the system.
"""

__all__ = ["DefaultObservation", "Observation"]

from .base import Observation
from .default import DefaultObservation

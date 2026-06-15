"""
Distributions module for scheduling instance generation.

This module provides specialized distributions for creating realistic scheduling
instances. It includes a base class for all distributions and several canonical
distributions commonly used in scheduling research.
"""

__all__ = [
    "Bernoulli",
    "BernoulliPrecedence",
    "Beta",
    "Categorical",
    "Choice",
    "DeterministicJobAssignment",
    "Dirichlet",
    "Distribution",
    "Exponential",
    "Geometric",
    "JobAssignmentProcess",
    "JobPartitionProcess",
    "Multinomial",
    "Normal",
    "Poisson",
    "PoissonProcess",
    "Process",
    "Range",
    "RejectionSampler",
    "Sampler",
    "Shuffled",
    "Uniform",
    "UniformInt",
    "UniformMachineEligibility",
]

from .base import (
    Distribution,
    JobPartitionProcess,
    Process,
    RejectionSampler,
    Sampler,
    Shuffled,
)
from .continuous import (
    Beta,
    Dirichlet,
    Exponential,
    Normal,
    Uniform,
)
from .deterministic import (
    DeterministicJobAssignment,
    Range,
)
from .discrete import (
    Bernoulli,
    Categorical,
    Choice,
    Geometric,
    Multinomial,
    Poisson,
    UniformInt,
)
from .scheduling import (
    BernoulliPrecedence,
    JobAssignmentProcess,
    PoissonProcess,
    UniformMachineEligibility,
)

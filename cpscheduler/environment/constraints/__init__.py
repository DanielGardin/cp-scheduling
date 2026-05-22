"""
This module defines the base class for all constraints in the scheduling environment.
It provides a common interface for any piece in the scheduling environment that
interacts with the tasks by limiting when they can be executed, how they are assigned to
machines, etc.

You can define your own constraints by subclassing the `Constraint` class and
implementing the required methods.

"""

__all__ = [
    "BatchConstraint",
    "ConstantProcessingTime",
    "Constraint",
    "DeadlineConstraint",
    "HorizonConstraint",
    "MachineBreakdownConstraint",
    "MachineConstraint",
    "MachineEligibilityConstraint",
    "NoWaitConstraint",
    "NonOverlapConstraint",
    "NonRenewableResourceConstraint",
    "ORPrecedenceConstraint",
    "OptionalityConstraint",
    "PassiveConstraint",
    "PrecedenceConstraint",
    "PreemptionConstraint",
    "ReleaseDateConstraint",
    "ResourceConstraint",
    "SetupConstraint",
    "constraints",
]

from .base import (
    Constraint,
    PassiveConstraint,
    constraints,
)
from .disjunctive import NonOverlapConstraint
from .machine import (
    BatchConstraint,
    MachineBreakdownConstraint,
    MachineConstraint,
)
from .passive import (
    ConstantProcessingTime,
    MachineEligibilityConstraint,
    OptionalityConstraint,
    PreemptionConstraint,
)
from .precedence import (
    NoWaitConstraint,
    ORPrecedenceConstraint,
    PrecedenceConstraint,
)
from .resources import (
    NonRenewableResourceConstraint,
    ResourceConstraint,
)
from .setup import SetupConstraint
from .time_windows import (
    DeadlineConstraint,
    HorizonConstraint,
    ReleaseDateConstraint,
)

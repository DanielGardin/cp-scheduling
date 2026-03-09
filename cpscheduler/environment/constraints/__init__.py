"""
This module defines the base class for all constraints in the scheduling environment.
It provides a common interface for any piece in the scheduling environment that
interacts with the tasks by limiting when they can be executed, how they are assigned to
machines, etc.

You can define your own constraints by subclassing the `Constraint` class and
implementing the required methods.

"""

__all__ = [
    "constraints",
    "Constraint",
    "PassiveConstraint",
    "MachineConstraint",
    "PreemptionConstraint",
    "OptionalityConstraint",
    "MachineEligibilityConstraint",
    "PrecedenceConstraint",
    "NoWaitConstraint",
    "ConstantProcessingTime",
    "NonOverlapConstraint",
    "ReleaseDateConstraint",
    "DeadlineConstraint",
    "HorizonConstraint",
    "ResourceConstraint",
    "NonRenewableResourceConstraint",
    "SetupConstraint",
    "MachineBreakdownConstraint",
]

from .base import (
    constraints,
    Constraint,
    PassiveConstraint,
)

from .machine import (
    MachineConstraint,
    MachineEligibilityConstraint,
    MachineBreakdownConstraint,
)

from .passive import (
    PreemptionConstraint,
    OptionalityConstraint,
    ConstantProcessingTime,
)

from .precedence import PrecedenceConstraint, NoWaitConstraint

from .resources import (
    ResourceConstraint,
    NonRenewableResourceConstraint,
)

from .setup import SetupConstraint

from .time_windows import (
    ReleaseDateConstraint,
    DeadlineConstraint,
    HorizonConstraint,
)

from .groups import NonOverlapConstraint

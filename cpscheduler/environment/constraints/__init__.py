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
    "ResourceConstraint",
    "NonRenewableResourceConstraint",
    "SetupConstraint",
    "MachineBreakdownConstraint",
]

from cpscheduler.environment.constraints.base import (
    constraints,
    Constraint,
    PassiveConstraint,
)

from cpscheduler.environment.constraints.machine import (
    MachineConstraint,
    MachineEligibilityConstraint,
    MachineBreakdownConstraint
)

from cpscheduler.environment.constraints.passive import (
    PreemptionConstraint,
    OptionalityConstraint,
    ConstantProcessingTime
)

from cpscheduler.environment.constraints.precedence import (
    PrecedenceConstraint,
    NoWaitConstraint
)

from cpscheduler.environment.constraints.resources import (
    ResourceConstraint,
    NonRenewableResourceConstraint,
)

from cpscheduler.environment.constraints.setup import SetupConstraint

from cpscheduler.environment.constraints.time_windows import (
    ReleaseDateConstraint,
    DeadlineConstraint,
)

from cpscheduler.environment.constraints.groups import NonOverlapConstraint

"""Module for defining scheduling constraints.

Constraints are beta componentes inside the environment, they interact with the
environment by changing the domains inside the constraint propagator kernel.
In high-level, constraints define how tasks interact with all other environment
features, such as machines, other tasks, resources, etc.

You can define your own constraints by subclassing the `Constraint` class and
implementing the required methods.
"""

__all__ = [
    "AtMostOneConstraint",
    "BatchConstraint",
    "ConstantProcessingTime",
    "Constraint",
    "DeadlineConstraint",
    "ExactlyOneConstraint",
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
    "RejectableConstraint",
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
from .optional import (
    AtMostOneConstraint,
    ExactlyOneConstraint,
    RejectableConstraint,
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

from typing import Any

from functools import singledispatch

from cpscheduler.environment._common import InstanceConfig
from cpscheduler.environment.constraints import (
    Constraint,
    PrecedenceConstraint,
    DisjunctiveConstraint,
    ReleaseDateConstraint,
    DeadlineConstraint,
    ResourceConstraint,
    MachineConstraint,
    SetupConstraint,
)


@singledispatch
def generate_constraint(
    setup: Constraint, instance: InstanceConfig, **kwargs: Any
) -> None:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


@generate_constraint.register
def _(constraint: PrecedenceConstraint, instance: InstanceConfig) -> None: ...

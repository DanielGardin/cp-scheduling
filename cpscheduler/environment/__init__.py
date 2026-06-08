"""Main environment class and all the components used to define a scheduling problem.

An environment is defined as a triplet of (alpha, beta, gamma), which corresponds
to the following components:

1. Alpha setups: These define the structure of the scheduling problem, such as
    the number of machines, the number of jobs, and the processing times.

2. Beta constraints: These define the constraints that must be satisfied in the
    scheduling problem, such as precedence constraints, machine eligibility
    constraints, and resource constraints.

3. Gamma objectives: These define the objective function that we want to optimize,
    such as makespan, total completion time, and total tardiness.

Examples
--------
>>> SchedulingEnv(JobShopSetup(), objective=Makespan())
SchedulingEnv(Jm||C_max, n_tasks=0)

>>> SchedulingEnv(SingleMachineSetup(), [ReleaseDateConstraint()], WeightedCompletionTime())
SchedulingEnv(1|r_j|Σw_jC_j, n_tasks=0)

"""

__all__ = [  # noqa: RUF022
    "SchedulingEnv",
    # Alpha setups
    "ScheduleSetup",
    "SingleMachineSetup",
    "IdenticalParallelMachineSetup",
    "UniformParallelMachineSetup",
    "UnrelatedParallelMachineSetup",
    "JobShopSetup",
    "FlexibleJobShopSetup",
    "FlowShopSetup",
    "OpenShopSetup",
    # Beta constraints
    "Constraint",
    "PassiveConstraint",
    "MachineConstraint",
    "PreemptionConstraint",
    "OptionalityConstraint",
    "MachineEligibilityConstraint",
    "PrecedenceConstraint",
    "NoWaitConstraint",
    "ORPrecedenceConstraint",
    "ConstantProcessingTime",
    "NonOverlapConstraint",
    "ReleaseDateConstraint",
    "DeadlineConstraint",
    "HorizonConstraint",
    "ResourceConstraint",
    "NonRenewableResourceConstraint",
    "SetupConstraint",
    "MachineBreakdownConstraint",
    "BatchConstraint",
    # Gamma objectives
    "Objective",
    "RegularObjective",
    "CompletionTimeObjective",
    "ComposedObjective",
    "Makespan",
    "TotalCompletionTime",
    "WeightedCompletionTime",
    "MaximumLateness",
    "TotalTardiness",
    "WeightedTardiness",
    "TotalEarliness",
    "WeightedEarliness",
    "TotalTardyJobs",
    "WeightedTardyJobs",
    "TotalFlowTime",
    "DiscountedTotalCompletionTime",
]

from .constraints import (
    BatchConstraint,
    ConstantProcessingTime,
    Constraint,
    DeadlineConstraint,
    HorizonConstraint,
    MachineBreakdownConstraint,
    MachineConstraint,
    MachineEligibilityConstraint,
    NonOverlapConstraint,
    NonRenewableResourceConstraint,
    NoWaitConstraint,
    OptionalityConstraint,
    ORPrecedenceConstraint,
    PassiveConstraint,
    PrecedenceConstraint,
    PreemptionConstraint,
    ReleaseDateConstraint,
    ResourceConstraint,
    SetupConstraint,
)
from .env import SchedulingEnv
from .objectives import (
    CompletionTimeObjective,
    ComposedObjective,
    DiscountedTotalCompletionTime,
    Makespan,
    MaximumLateness,
    Objective,
    RegularObjective,
    TotalCompletionTime,
    TotalEarliness,
    TotalFlowTime,
    TotalTardiness,
    TotalTardyJobs,
    WeightedCompletionTime,
    WeightedEarliness,
    WeightedTardiness,
    WeightedTardyJobs,
)
from .setups import (
    FlexibleJobShopSetup,
    FlowShopSetup,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    OpenShopSetup,
    ScheduleSetup,
    SingleMachineSetup,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
)

"""
This module contains the main environment class and all the components that are
used to define a scheduling problem.
An environment is defined as a triplet of (alpha, beta, gamma), which corresponds
to the following components:

1. Alpha setups: These define the structure of the scheduling problem, such as
    the number of machines, the number of jobs, and the processing times.

2. Beta constraints: These define the constraints that must be satisfied in the
    scheduling problem, such as precedence constraints, machine eligibility
    constraints, and resource constraints.

3. Gamma objectives: These define the objective function that we want to optimize,
    such as makespan, total completion time, and total tardiness.
"""

__all__ = [
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
    "SoftConstraint",
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
    # Gamma objectives
    "Objective",
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
    "DiscountedCompletionTime",
    "RejectionCost",
]

from .env import SchedulingEnv

from .schedule_setup import *
from .constraints import *
from .objectives import *

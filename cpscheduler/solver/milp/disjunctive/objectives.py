from pulp import lpSum, lpDot

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.objectives import (
    Objective,
    ComposedObjective,
    Makespan,
    TotalCompletionTime,
    WeightedCompletionTime,
    MaximumLateness,
    TotalTardiness,
    WeightedTardiness,
    TotalEarliness,
    WeightedEarliness,
    TotalTardyJobs,
    WeightedTardyJobs,
    TotalFlowTime,
)

from cpscheduler.solver.cp.utils import scale_to_int
from cpscheduler.solver.milp.pulp_utils import PULP_PARAM, max_pulp

from cpscheduler.solver.milp.disjunctive.formulation import (
    DisjunctiveMILPFormulation,
)


def jobs_makespan(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
) -> list[PULP_PARAM]:
    makespans = []

    for job_id, job_tasks in enumerate(state.instance.job_tasks):
        job_makespan = max_pulp(
            formulation.model,
            [formulation.end_times[task_id] for task_id in job_tasks],
            name=f"makespan_job_{job_id}",
        )
        makespans.append(job_makespan)

    return makespans


@DisjunctiveMILPFormulation.register_objective(Objective)
def objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: Objective,
) -> int:
    return 0


@DisjunctiveMILPFormulation.register_objective(Makespan)
def makespan_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: Makespan,
) -> PULP_PARAM:
    makespan = max_pulp(
        formulation.model,
        formulation.end_times,
        name="makespan",
    )

    formulation.model.setObjective(makespan)

    return makespan


@DisjunctiveMILPFormulation.register_objective(ComposedObjective)
def composed_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: ComposedObjective,
) -> PULP_PARAM:
    sub_objectives = [
        DisjunctiveMILPFormulation._objective_registry[type(sub_objective)](
            formulation, state, sub_objective
        )
        for sub_objective in objective.objectives
    ]

    coefficients = objective.coefficients
    objective_value = lpDot(coefficients, sub_objectives)

    formulation.model.setObjective(objective_value)

    return objective_value


@DisjunctiveMILPFormulation.register_objective(TotalCompletionTime)
def total_completion_time_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalCompletionTime,
) -> PULP_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    total_completion_time = lpSum(job_makespans)

    formulation.model.setObjective(total_completion_time)

    return total_completion_time


@DisjunctiveMILPFormulation.register_objective(WeightedCompletionTime)
def weighted_completion_time_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedCompletionTime,
) -> PULP_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    weights = (
        scale_to_int(objective.job_weights)
        if formulation.relaxed
        else objective.job_weights
    )

    weighted_completion_time = lpDot(weights, job_makespans)

    formulation.model.setObjective(weighted_completion_time)

    return weighted_completion_time

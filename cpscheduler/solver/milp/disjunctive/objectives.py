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
    TotalFlowTime
)

from cpscheduler.solver.milp.disjunctive.formulation import (
    DisjunctiveMILPFormulation,
)
from cpscheduler.solver.milp.pyomo_formulation import PYOMO_PARAM

def jobs_makespan(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
) -> list[PYOMO_PARAM]:
    makespans = []

    for job_id, job_tasks in enumerate(state.instance.job_tasks):
        job_makespan = formulation.max_expr(
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
    formulation.set_objective(0)
    return 0


@DisjunctiveMILPFormulation.register_objective(Makespan)
def makespan_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: Makespan,
) -> PYOMO_PARAM:
    makespan = formulation.max_expr(
        formulation.end_times,
        name="makespan",
    )

    formulation.set_objective(makespan)

    return makespan


@DisjunctiveMILPFormulation.register_objective(ComposedObjective)
def composed_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: ComposedObjective,
) -> PYOMO_PARAM:
    sub_objectives = [
        DisjunctiveMILPFormulation._objective_registry[type(sub_objective)](
            formulation, state, sub_objective
        )
        for sub_objective in objective.objectives
    ]

    coefficients = objective.coefficients
    objective_value = sum(
        coefficient * sub_objective
        for coefficient, sub_objective in zip(coefficients, sub_objectives)
    )

    formulation.set_objective(objective_value)

    return objective_value


@DisjunctiveMILPFormulation.register_objective(TotalCompletionTime)
def total_completion_time_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalCompletionTime,
) -> PYOMO_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    total_completion_time = sum(job_makespans)

    formulation.set_objective(total_completion_time)

    return total_completion_time


@DisjunctiveMILPFormulation.register_objective(WeightedCompletionTime)
def weighted_completion_time_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedCompletionTime,
) -> PYOMO_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    weights = objective.job_weights

    weighted_completion_time = sum(
        weight * makespan for weight, makespan in zip(weights, job_makespans)
    )

    formulation.set_objective(weighted_completion_time)

    return weighted_completion_time


@DisjunctiveMILPFormulation.register_objective(MaximumLateness)
def maximum_lateness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: MaximumLateness,
) -> PYOMO_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    lateness = [
        job_makespan - int(due_date)
        for job_makespan, due_date in zip(job_makespans, objective.due_dates)
    ]
    max_lateness = formulation.max_expr(lateness, name="max_lateness")

    formulation.set_objective(max_lateness)

    return max_lateness


@DisjunctiveMILPFormulation.register_objective(TotalTardiness)
def total_tardiness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalTardiness,
) -> PYOMO_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    tardiness_terms = [
        formulation.max_expr(
            [0, job_makespan - int(due_date)],
            name=f"tardiness_{job_id}",
        )
        for job_id, (job_makespan, due_date) in enumerate(
            zip(job_makespans, objective.due_dates)
        )
    ]
    total_tardiness = sum(tardiness_terms)

    formulation.set_objective(total_tardiness)

    return total_tardiness


@DisjunctiveMILPFormulation.register_objective(WeightedTardiness)
def weighted_tardiness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedTardiness,
) -> PYOMO_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    weights = objective.job_weights

    terms = []
    for job_id, (job_makespan, due_date, weight) in enumerate(
        zip(job_makespans, objective.due_dates, weights)
    ):
        if weight == 0:
            continue

        tardiness = formulation.max_expr(
            [0, job_makespan - int(due_date)],
            name=f"weighted_tardiness_{job_id}",
        )
        terms.append(weight * tardiness)

    weighted_tardiness = sum(terms)

    formulation.set_objective(weighted_tardiness)

    return weighted_tardiness


@DisjunctiveMILPFormulation.register_objective(TotalEarliness)
def total_earliness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalEarliness,
) -> PYOMO_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    earliness_terms = [
        formulation.max_expr(
            [0, int(due_date) - job_makespan],
            name=f"earliness_{job_id}",
        )
        for job_id, (job_makespan, due_date) in enumerate(
            zip(job_makespans, objective.due_dates)
        )
    ]
    total_earliness = sum(earliness_terms)

    formulation.set_objective(total_earliness)

    return total_earliness


@DisjunctiveMILPFormulation.register_objective(WeightedEarliness)
def weighted_earliness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedEarliness,
) -> PYOMO_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    weights = objective.job_weights

    terms = []
    for job_id, (job_makespan, due_date, weight) in enumerate(
        zip(job_makespans, objective.due_dates, weights)
    ):
        if weight == 0:
            continue

        earliness = formulation.max_expr(
            [0, int(due_date) - job_makespan],
            name=f"weighted_earliness_{job_id}",
        )
        terms.append(weight * earliness)

    weighted_earliness = sum(terms)

    formulation.set_objective(weighted_earliness)

    return weighted_earliness


@DisjunctiveMILPFormulation.register_objective(TotalTardyJobs)
def total_tardy_jobs_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalTardyJobs,
) -> PYOMO_PARAM:

    tardy_indicators = [
        formulation.add_var(
            f"tardy_{job_id}",
            binary=True
        )
        for job_id in range(state.n_jobs)
    ]

    task_ids = state.instance.job_tasks
    for job_id, due_date in enumerate(objective.due_dates):
        U_j = tardy_indicators[job_id]

        for task_id in task_ids[job_id]:
            C_j = formulation.end_times[task_id]

            formulation.implication(
                (1 - U_j,),
                (C_j, '<=', due_date),
                f"tardy_implication_{task_id}"
            )

    total_tardy_jobs = sum(tardy_indicators)

    formulation.set_objective(total_tardy_jobs)

    return total_tardy_jobs


@DisjunctiveMILPFormulation.register_objective(WeightedTardyJobs)
def weighted_tardy_jobs_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedTardyJobs,
) -> PYOMO_PARAM:

    tardy_indicators = [
        formulation.add_var(
            f"tardy_{job_id}",
            binary=True
        )
        for job_id in range(state.n_jobs)
    ]

    task_ids = state.instance.job_tasks
    for job_id, due_date in enumerate(objective.due_dates):
        U_j = tardy_indicators[job_id]

        for task_id in task_ids[job_id]:
            C_j = formulation.end_times[task_id]

            formulation.implication(
                (1 - U_j,),
                (C_j, '<=', due_date),
                f"tardy_implication_{task_id}"
            )

    weighted_tardy_jobs = sum(
        tardy_indicators[job_id] * weight
        for job_id, weight in enumerate(objective.job_weights)
    )

    formulation.set_objective(weighted_tardy_jobs)

    return weighted_tardy_jobs


@DisjunctiveMILPFormulation.register_objective(TotalFlowTime)
def total_flow_time_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalFlowTime,
) -> PYOMO_PARAM:
    job_makespans = jobs_makespan(formulation, state)

    flow_terms = [
        formulation.max_expr(
            [0, job_makespan - int(release_time)],
            name=f"flow_time_{job_id}",
        )
        for job_id, (job_makespan, release_time) in enumerate(
            zip(job_makespans, objective.release_times)
        )
    ]
    total_flow_time = sum(flow_terms)

    formulation.set_objective(total_flow_time)

    return total_flow_time


# @DisjunctiveMILPFormulation.register_objective(RejectionCost)
# def rejection_cost_objective(
#     formulation: DisjunctiveMILPFormulation,
#     state: ScheduleState,
#     objective: RejectionCost,
# ) -> PYOMO_PARAM:
#     weights =  objective.rejection_costs

#     terms: list[PYOMO_PARAM] = []
#     for task_id, optional in enumerate(state.instance.optional):
#         if not optional:
#             continue

#         cost = weights[task_id]
#         if cost == 0:
#             continue

#         present = formulation.present[task_id]
#         terms.append(cost * (1 - present))

#     rejection_cost = sum(terms)

#     formulation.set_objective(rejection_cost)

#     return rejection_cost

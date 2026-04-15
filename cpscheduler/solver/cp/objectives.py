from collections.abc import Iterable

from cpscheduler.environment import (
    ComposedObjective,
    Makespan,
    MaximumLateness,
    Objective,
    TotalCompletionTime,
    TotalEarliness,
    TotalFlowTime,
    TotalTardyJobs,
    TotalTardiness,
    WeightedCompletionTime,
    WeightedEarliness,
    WeightedTardyJobs,
    WeightedTardiness,
)
from cpscheduler.environment.state import ScheduleState

from .cp_formulation import DisjunctiveCPFormulation


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))

    return str(value)


def _job_completion_expr(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    job_id: int,
) -> str:
    task_ids = state.instance.job_tasks[job_id]
    completion = formulation.max_expr(
        [formulation._end_expr(task_id) for task_id in task_ids],
        name=f"job_completion_{job_id}",
        ub=formulation._horizon,
    )
    return completion.name


def _job_completion_exprs(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
) -> list[str]:
    return [
        _job_completion_expr(formulation, state, job_id)
        for job_id in range(state.n_jobs)
    ]


def _sum_terms(terms: Iterable[str]) -> str:
    filtered = [term for term in terms if term != "0"]
    return " + ".join(filtered) if filtered else "0"


@DisjunctiveCPFormulation.register_objective(Objective)
def objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: Objective,
) -> int:
    formulation.set_objective(0)
    return 0


@DisjunctiveCPFormulation.register_objective(Makespan)
def makespan_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: Makespan,
) -> str:
    makespan = formulation.max_expr(
        [formulation._end_expr(task_id) for task_id in range(state.n_tasks)],
        name="makespan",
        ub=formulation._horizon,
    )
    formulation.set_objective(makespan.name)
    return makespan.name


@DisjunctiveCPFormulation.register_objective(ComposedObjective)
def composed_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: ComposedObjective,
) -> str:
    sub_objectives = [
        DisjunctiveCPFormulation._objective_registry[type(sub_objective)](
            formulation, state, sub_objective
        )
        for sub_objective in objective.objectives
    ]

    terms: list[str] = []
    for coefficient, sub_objective in zip(objective.coefficients, sub_objectives):
        if coefficient == 0:
            continue

        coef = _format_number(coefficient)
        if coef == "1":
            terms.append(sub_objective)
        elif coef == "-1":
            terms.append(f"- {sub_objective}")
        else:
            terms.append(f"{coef} * {sub_objective}")

    objective_expr = _sum_terms(terms)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(TotalCompletionTime)
def total_completion_time_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: TotalCompletionTime,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    objective_expr = _sum_terms(job_completion)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(WeightedCompletionTime)
def weighted_completion_time_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: WeightedCompletionTime,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    terms = [
        f"{_format_number(weight)} * {completion}"
        for weight, completion in zip(objective.job_weights, job_completion)
        if weight != 0
    ]
    objective_expr = _sum_terms(terms)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(MaximumLateness)
def maximum_lateness_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: MaximumLateness,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    lateness = [
        f"{completion} - {due_date}"
        for completion, due_date in zip(job_completion, objective.due_dates)
    ]
    max_lateness = formulation.max_expr(lateness, name="max_lateness")
    formulation.set_objective(max_lateness.name)
    return max_lateness.name


@DisjunctiveCPFormulation.register_objective(TotalTardiness)
def total_tardiness_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: TotalTardiness,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    tardiness_terms = [
        formulation.max_expr(
            [0, f"{completion} - {due_date}"],
            name=f"tardiness_{job_id}",
            ub=formulation._horizon,
        ).name
        for job_id, (completion, due_date) in enumerate(
            zip(job_completion, objective.due_dates)
        )
    ]
    objective_expr = _sum_terms(tardiness_terms)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(WeightedTardiness)
def weighted_tardiness_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: WeightedTardiness,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    terms = []
    for job_id, (completion, due_date, weight) in enumerate(
        zip(job_completion, objective.due_dates, objective.job_weights)
    ):
        tardiness = formulation.max_expr(
            [0, f"{completion} - {due_date}"],
            name=f"weighted_tardiness_{job_id}",
            ub=formulation._horizon,
        ).name
        if weight != 0:
            terms.append(f"{_format_number(weight)} * {tardiness}")

    objective_expr = _sum_terms(terms)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(TotalEarliness)
def total_earliness_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: TotalEarliness,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    earliness_terms = [
        formulation.max_expr(
            [0, f"{due_date} - {completion}"],
            name=f"earliness_{job_id}",
            ub=formulation._horizon,
        ).name
        for job_id, (completion, due_date) in enumerate(
            zip(job_completion, objective.due_dates)
        )
    ]
    objective_expr = _sum_terms(earliness_terms)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(WeightedEarliness)
def weighted_earliness_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: WeightedEarliness,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    terms = []
    for job_id, (completion, due_date, weight) in enumerate(
        zip(job_completion, objective.due_dates, objective.job_weights)
    ):
        earliness = formulation.max_expr(
            [0, f"{due_date} - {completion}"],
            name=f"weighted_earliness_{job_id}",
            ub=formulation._horizon,
        ).name
        if weight != 0:
            terms.append(f"{_format_number(weight)} * {earliness}")

    objective_expr = _sum_terms(terms)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(TotalTardyJobs)
def total_tardy_jobs_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: TotalTardyJobs,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    tardy_terms = [
        f"bool2int({completion} > {due_date})"
        for completion, due_date in zip(job_completion, objective.due_dates)
    ]
    objective_expr = _sum_terms(tardy_terms)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(WeightedTardyJobs)
def weighted_tardy_jobs_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: WeightedTardyJobs,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    terms = [
        f"{_format_number(weight)} * bool2int({completion} > {due_date})"
        for completion, due_date, weight in zip(
            job_completion, objective.due_dates, objective.job_weights
        )
        if weight != 0
    ]
    objective_expr = _sum_terms(terms)
    formulation.set_objective(objective_expr)
    return objective_expr


@DisjunctiveCPFormulation.register_objective(TotalFlowTime)
def total_flow_time_objective(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    objective: TotalFlowTime,
) -> str:
    job_completion = _job_completion_exprs(formulation, state)
    flow_terms = [
        formulation.max_expr(
            [0, f"{completion} - {release_time}"],
            name=f"flow_time_{job_id}",
            ub=formulation._horizon,
        ).name
        for job_id, (completion, release_time) in enumerate(
            zip(job_completion, objective.release_times)
        )
    ]
    objective_expr = _sum_terms(flow_terms)
    formulation.set_objective(objective_expr)
    return objective_expr


# @DisjunctiveCPFormulation.register_objective(RejectionCost)
# def rejection_cost_objective(
#     formulation: DisjunctiveCPFormulation,
#     state: ScheduleState,
#     objective: RejectionCost,
# ) -> str:
#     terms = []
#     for task_id, optional in enumerate(state.instance.optional):
#         if not optional:
#             continue

#         cost = objective.rejection_costs[task_id]
#         if cost == 0:
#             continue

#         present = formulation._expr(formulation.presents[task_id])
#         terms.append(f"{_format_number(cost)} * (1 - bool2int({present}))")

#     objective_expr = _sum_terms(terms)
#     formulation.set_objective(objective_expr)
#     return objective_expr

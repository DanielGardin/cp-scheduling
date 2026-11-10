"""Module for defining objective functions for the disjunctive MILP formulation."""

from pyomo.environ import quicksum

from cpscheduler.environment.objectives import (
    ComposedObjective,
    Makespan,
    MaximumLateness,
    RejectionCost,
    SatisfactionObjective,
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
from cpscheduler.environment.state import ScheduleState
from cpscheduler.solver.milp.disjunctive.formulation import (
    DisjunctiveMILPFormulation,
)
from cpscheduler.solver.milp.pyomo_formulation import PYOMO_PARAM


def _jobs_makespan(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    exact: bool = False,
) -> list[PYOMO_PARAM]:
    """Return variables for the job completion times.

    A non-exact variable is only an upper bound of the completion time, which
    is tight solely when the objective pushes it down. Objectives improved by
    later completions, like earliness, must ask for exact variables.

    Absent tasks do not count, and a job without present tasks completes at 0.
    """
    makespans = []

    for job_id, job_tasks in enumerate(state.instance.job_tasks):
        job_makespan = formulation.max_expr(
            [formulation.end_times[task_id] for task_id in job_tasks],
            name=f"makespan_job_{job_id}",
            exact=exact,
            conditions=[formulation.present[task_id] for task_id in job_tasks],
        )
        makespans.append(job_makespan)

    return makespans


def _jobs_presence(
    formulation: DisjunctiveMILPFormulation, state: ScheduleState
) -> list[PYOMO_PARAM]:
    """Return whether each job has a present task."""
    return [
        formulation.job_presence(job_id, job_tasks)
        for job_id, job_tasks in enumerate(state.instance.job_tasks)
    ]


@DisjunctiveMILPFormulation.register_objective(SatisfactionObjective)
def _objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: SatisfactionObjective,
) -> int:
    formulation.set_objective(0)
    return 0


@DisjunctiveMILPFormulation.register_objective(Makespan)
def _makespan_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: Makespan,
) -> PYOMO_PARAM:
    makespan = formulation.max_expr(
        formulation.end_times,
        name="makespan",
        conditions=formulation.present,
    )

    formulation.set_objective(makespan)

    return makespan


@DisjunctiveMILPFormulation.register_objective(ComposedObjective)
def _composed_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: ComposedObjective,
) -> PYOMO_PARAM:
    sub_objectives: list[PYOMO_PARAM] = [
        DisjunctiveMILPFormulation.get_objective_fn(type(sub_objective))(
            formulation, state, sub_objective
        )
        for sub_objective in objective.objectives
    ]

    coefficients = objective.coefficients
    objective_value = quicksum(
        coefficient * sub_objective
        for coefficient, sub_objective in zip(
            coefficients, sub_objectives, strict=False
        )
    )

    formulation.set_objective(objective_value)

    return objective_value


@DisjunctiveMILPFormulation.register_objective(TotalCompletionTime)
def _total_completion_time_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalCompletionTime,
) -> PYOMO_PARAM:
    job_makespans = _jobs_makespan(formulation, state)

    total_completion_time = quicksum(job_makespans)

    formulation.set_objective(total_completion_time)

    return total_completion_time


@DisjunctiveMILPFormulation.register_objective(WeightedCompletionTime)
def _weighted_completion_time_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedCompletionTime,
) -> PYOMO_PARAM:
    job_makespans = _jobs_makespan(formulation, state)

    weights = objective.weights.value

    weighted_completion_time = quicksum(
        weight * makespan
        for weight, makespan in zip(weights, job_makespans, strict=False)
    )

    formulation.set_objective(weighted_completion_time)

    return weighted_completion_time


@DisjunctiveMILPFormulation.register_objective(MaximumLateness)
def _maximum_lateness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: MaximumLateness,
) -> PYOMO_PARAM:
    job_makespans = _jobs_makespan(formulation, state)
    due_dates = [int(due_date) for due_date in objective.due_dates.value]
    presence = _jobs_presence(formulation, state)

    lateness = [
        job_makespan - due_date
        for job_makespan, due_date in zip(
            job_makespans, due_dates, strict=False
        )
    ]

    # A rejected job is left out of the maximum, which is 0 without any job
    max_lateness = formulation.max_expr(
        lateness, name="max_lateness", conditions=presence
    )

    formulation.set_objective(max_lateness)

    return max_lateness


@DisjunctiveMILPFormulation.register_objective(TotalTardiness)
def _total_tardiness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalTardiness,
) -> PYOMO_PARAM:
    job_makespans = _jobs_makespan(formulation, state)

    due_dates = objective.due_dates.value

    tardiness_terms = [
        formulation.max_expr(
            [0, job_makespan - int(due_date)],
            name=f"tardiness_{job_id}",
        )
        for job_id, (job_makespan, due_date) in enumerate(
            zip(job_makespans, due_dates, strict=False)
        )
    ]
    total_tardiness = quicksum(tardiness_terms)

    formulation.set_objective(total_tardiness)

    return total_tardiness


@DisjunctiveMILPFormulation.register_objective(WeightedTardiness)
def _weighted_tardiness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedTardiness,
) -> PYOMO_PARAM:
    job_makespans = _jobs_makespan(formulation, state)

    weights = objective.weights.value
    due_dates = objective.due_dates.value

    terms = []
    for job_id, (job_makespan, due_date, weight) in enumerate(
        zip(job_makespans, due_dates, weights, strict=False)
    ):
        if weight == 0:
            continue

        tardiness = formulation.max_expr(
            [0, job_makespan - int(due_date)],
            name=f"weighted_tardiness_{job_id}",
        )
        terms.append(weight * tardiness)

    weighted_tardiness = quicksum(terms)

    formulation.set_objective(weighted_tardiness)

    return weighted_tardiness


@DisjunctiveMILPFormulation.register_objective(TotalEarliness)
def _total_earliness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalEarliness,
) -> PYOMO_PARAM:
    job_makespans = _jobs_makespan(formulation, state, exact=True)
    presence = _jobs_presence(formulation, state)

    due_dates = objective.due_dates.value

    # A rejected job completes at 0, its due date must not count either
    earliness_terms = [
        formulation.max_expr(
            [0, int(due_date) * present - job_makespan],
            name=f"earliness_{job_id}",
        )
        for job_id, (job_makespan, due_date, present) in enumerate(
            zip(job_makespans, due_dates, presence, strict=False)
        )
    ]
    total_earliness = quicksum(earliness_terms)

    formulation.set_objective(total_earliness)

    return total_earliness


@DisjunctiveMILPFormulation.register_objective(WeightedEarliness)
def _weighted_earliness_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedEarliness,
) -> PYOMO_PARAM:
    job_makespans = _jobs_makespan(formulation, state, exact=True)
    presence = _jobs_presence(formulation, state)

    weights = objective.weights.value
    due_dates = objective.due_dates.value

    terms = []
    for job_id, (job_makespan, due_date, weight, present) in enumerate(
        zip(job_makespans, due_dates, weights, presence, strict=False)
    ):
        if weight == 0:
            continue

        earliness = formulation.max_expr(
            [0, int(due_date) * present - job_makespan],
            name=f"weighted_earliness_{job_id}",
        )
        terms.append(weight * earliness)

    weighted_earliness = quicksum(terms)

    formulation.set_objective(weighted_earliness)

    return weighted_earliness


@DisjunctiveMILPFormulation.register_objective(TotalTardyJobs)
def _total_tardy_jobs_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalTardyJobs,
) -> PYOMO_PARAM:
    tardy_indicators = [
        formulation.add_var(f"tardy_{job_id}", binary=True)
        for job_id in range(state.n_jobs)
    ]

    task_ids = state.instance.job_tasks
    due_dates = objective.due_dates.value

    for job_id, due_date in enumerate(due_dates):
        U_j = tardy_indicators[job_id]

        for task_id in task_ids[job_id]:
            # An absent task does not make its job tardy
            formulation.implication(
                (1 - U_j, formulation.present[task_id]),
                (formulation.end_times[task_id], "<=", due_date),
                f"tardy_implication_{task_id}",
            )

    total_tardy_jobs = quicksum(tardy_indicators)

    formulation.set_objective(total_tardy_jobs)

    return total_tardy_jobs


@DisjunctiveMILPFormulation.register_objective(WeightedTardyJobs)
def _weighted_tardy_jobs_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: WeightedTardyJobs,
) -> PYOMO_PARAM:
    tardy_indicators = [
        formulation.add_var(f"tardy_{job_id}", binary=True)
        for job_id in range(state.n_jobs)
    ]

    task_ids = state.instance.job_tasks
    due_dates = objective.due_dates.value
    weights = objective.weights.value

    for job_id, due_date in enumerate(due_dates):
        U_j = tardy_indicators[job_id]

        for task_id in task_ids[job_id]:
            # An absent task does not make its job tardy
            formulation.implication(
                (1 - U_j, formulation.present[task_id]),
                (formulation.end_times[task_id], "<=", due_date),
                f"tardy_implication_{task_id}",
            )

    weighted_tardy_jobs = quicksum(
        tardy_indicators[job_id] * weight
        for job_id, weight in enumerate(weights)
    )

    formulation.set_objective(weighted_tardy_jobs)

    return weighted_tardy_jobs


@DisjunctiveMILPFormulation.register_objective(TotalFlowTime)
def _total_flow_time_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: TotalFlowTime,
) -> PYOMO_PARAM:
    job_makespans = _jobs_makespan(formulation, state)

    release_times = objective.release_times.value

    flow_terms = [
        formulation.max_expr(
            [0, job_makespan - int(release_time)],
            name=f"flow_time_{job_id}",
        )
        for job_id, (job_makespan, release_time) in enumerate(
            zip(job_makespans, release_times, strict=False)
        )
    ]
    total_flow_time = quicksum(flow_terms)

    formulation.set_objective(total_flow_time)

    return total_flow_time


@DisjunctiveMILPFormulation.register_objective(RejectionCost)
def _rejection_cost_objective(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    objective: RejectionCost,
) -> PYOMO_PARAM:
    # A job is rejected when none of its tasks is present
    rejection_cost = sum(
        cost * (1 - formulation.job_presence(job_id, tasks))
        for job_id, (cost, tasks) in enumerate(
            zip(
                objective.rejection_costs.value,
                state.instance.job_tasks,
                strict=True,
            )
        )
        if cost != 0 and tasks
    )

    formulation.set_objective(rejection_cost)

    return rejection_cost

from typing import Callable

from pulp import LpProblem, lpSum, LpVariable, LpInteger, lpDot

from functools import singledispatch

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

from .tasks import PulpVariables
from .utils import max_pulp, indicator_var

@singledispatch
def export_objective_pulp(setup: Objective) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


@export_objective_pulp.register
def _(objective: ComposedObjective) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        composed_objective = LpVariable("composed_objective")

        sub_objectives = [
            export_objective_pulp(sub_objective)(model, decision_vars)
            for sub_objective in objective.objectives
        ]

        model.addConstraint(
            composed_objective == lpDot(
                objective.coefficients,
                sub_objectives
            )
        )

        return composed_objective

    return export_model

@export_objective_pulp.register
def _(objective: Makespan) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        makespan = max_pulp(
            model,
            decision_vars.end_times,
            cat=LpInteger,
            name="makespan"
        )

        return makespan

    return export_model

@export_objective_pulp.register
def _(objective: TotalCompletionTime) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        jobs_makespan = [
            max_pulp(
                model,
                [decision_vars.end_times[task.task_id] for task in tasks],
                cat=LpInteger,
                name=f"job_{tasks[0]}_makespan"
            )
            for tasks in objective.tasks.jobs
        ]

        return lpSum(jobs_makespan)

    return export_model

@export_objective_pulp.register
def _(objective: WeightedCompletionTime) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        jobs_makespan = [
            max_pulp(
                model,
                [decision_vars.end_times[task.task_id] for task in tasks],
                cat=LpInteger,
                name=f"job_{tasks[0]}_makespan"
            )
            for tasks in objective.tasks.jobs
        ]

        return lpDot(objective.job_weights, jobs_makespan)

    return export_model

@export_objective_pulp.register
def _(objective: MaximumLateness) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        lateness = [
            end_time - due_date
            for end_time, due_date in zip(decision_vars.end_times, objective.due_dates)
        ]

        maximum_lateness = max_pulp(
            model,
            lateness,
            cat=LpInteger,
            name="makespan"
        )

        return maximum_lateness

    return export_model

@export_objective_pulp.register
def _(objective: TotalTardiness) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        jobs_tardiness = [
            max_pulp(
                model,
                [
                    decision_vars.end_times[task.task_id] - due_date for task in tasks
                ] + [0],
                cat=LpInteger,
                name=f"job_{tasks[0]}_makespan"
            )
            for tasks, due_date in zip(objective.tasks.jobs, objective.due_dates)
        ]

        return lpSum(jobs_tardiness)

    return export_model

@export_objective_pulp.register
def _(objective: WeightedTardiness) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        jobs_tardiness = [
            max_pulp(
                model,
                [
                    decision_vars.end_times[task.task_id] - due_date for task in tasks
                ] + [0],
                cat=LpInteger,
                name=f"job_{tasks[0]}_makespan"
            )
            for tasks, due_date in zip(objective.tasks.jobs, objective.due_dates)
        ]

        return lpDot(objective.job_weights, jobs_tardiness)

    return export_model

@export_objective_pulp.register
def _(objective: TotalEarliness) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        jobs_earliness = [
            max_pulp(
                model,
                [
                    due_date - decision_vars.end_times[task.task_id] for task in tasks
                ] + [0],
                cat=LpInteger,
                name=f"job_{tasks[0]}_makespan"
            )
            for tasks, due_date in zip(objective.tasks.jobs, objective.due_dates)
        ]

        return lpSum(jobs_earliness)

    return export_model

@export_objective_pulp.register
def _(objective: WeightedEarliness) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        jobs_earliness = [
            max_pulp(
                model,
                [
                    due_date - decision_vars.end_times[task.task_id] for task in tasks
                ] + [0],
                cat=LpInteger,
                name=f"job_{tasks[0]}_makespan"
            )
            for tasks, due_date in zip(objective.tasks.jobs, objective.due_dates)
        ]

        return lpDot(objective.job_weights, jobs_earliness)

    return export_model

@export_objective_pulp.register
def _(objective: TotalTardyJobs) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> LpVariable:
        tardy_jobs = 0

        for tasks, due_date in zip(objective.tasks.jobs, objective.due_dates):
            max_tardiness = max_pulp(
                model,
                [
                    decision_vars.end_times[task.task_id] - due_date for task in tasks
                ] + [0],
                cat=LpInteger,
                name=f"job_{tasks[0]}_tardiness"
            )

            tardy_jobs += indicator_var(
                model,
                lhs=max_tardiness,
                operator=">=",
                rhs=1,
                big_m=1
            )

        assert isinstance(tardy_jobs, LpVariable)
        return tardy_jobs

    return export_model

@export_objective_pulp.register
def _(objective: WeightedTardyJobs) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    raise NotImplementedError("WeightedTardyJobs objective is not implemented for PuLP.")

@export_objective_pulp.register
def _(objective: TotalFlowTime) -> Callable[[LpProblem, PulpVariables], LpVariable]:
    raise NotImplementedError("TotalFlowTime objective is not implemented for PuLP.")
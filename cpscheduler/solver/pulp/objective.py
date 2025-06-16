from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, lpSum, LpVariable, LpInteger, lpDot, LpAffineExpression

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

ObjectiveVar: TypeAlias = LpVariable | LpAffineExpression

@singledispatch
def export_objective_pulp(setup: Objective) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


@export_objective_pulp.register
def _(objective: ComposedObjective) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
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
def _(objective: Makespan) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
        makespan = max_pulp(
            model,
            decision_vars.end_times,
            cat=LpInteger,
            name="makespan"
        )

        return makespan

    return export_model

@export_objective_pulp.register
def _(objective: TotalCompletionTime) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
        jobs_makespan = [
            max_pulp(
                model,
                [decision_vars.end_times[task.task_id] for task in tasks],
                cat=LpInteger,
                name=f"job_{job_id}_makespan"
            )
            for job_id, tasks in enumerate(objective.tasks.jobs)
        ]

        return lpSum(jobs_makespan)

    return export_model

@export_objective_pulp.register
def _(objective: WeightedCompletionTime) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
        jobs_makespan = [
            max_pulp(
                model,
                [decision_vars.end_times[task.task_id] for task in tasks],
                cat=LpInteger,
                name=f"job_{job_id}_makespan"
            )
            for job_id, tasks in enumerate(objective.tasks.jobs)
        ]

        return lpDot(objective.job_weights, jobs_makespan)

    return export_model

@export_objective_pulp.register
def _(objective: MaximumLateness) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
        lateness = [
            end_time - due_date
            for end_time, due_date in zip(decision_vars.end_times, objective.due_dates)
        ]

        maximum_lateness = max_pulp(
            model,
            lateness,
            cat=LpInteger,
            name="maximum_lateness"
        )

        return maximum_lateness

    return export_model

@export_objective_pulp.register
def _(objective: TotalTardiness) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
        total_tardiness = 0
        for job_id, tasks in enumerate(objective.tasks.jobs):
            tardiness = [
                decision_vars.end_times[task.task_id] - objective.due_dates[job_id]
                for task in tasks
            ]

            max_tardiness = LpVariable(
                f"job_{job_id}_tardiness",
                lowBound=0,
                cat=LpInteger
            )

            total_tardiness += max_pulp(
                model,
                tardiness,
                max_tardiness,
                name=f"job_{job_id}_tardiness"
            )

        assert isinstance(total_tardiness, ObjectiveVar)
        return total_tardiness

    return export_model

@export_objective_pulp.register
def _(objective: WeightedTardiness) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
        weighted_tardiness = 0
        for job_id, tasks in enumerate(objective.tasks.jobs):
            tardiness = [
                decision_vars.end_times[task.task_id] - objective.due_dates[job_id]
                for task in tasks
            ]

            max_tardiness = LpVariable(
                f"job_{job_id}_tardiness",
                lowBound=0,
                cat=LpInteger
            )

            weighted_tardiness += objective.job_weights[job_id] * max_pulp(
                model,
                tardiness,
                max_tardiness,
                name=f"job_{job_id}_tardiness"
            )

        assert isinstance(weighted_tardiness, ObjectiveVar)
        return weighted_tardiness

    return export_model

@export_objective_pulp.register
def _(objective: TotalEarliness) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
        total_earliness = 0
        for job_id, tasks in enumerate(objective.tasks.jobs):
            earliness = [
                objective.due_dates[job_id] - decision_vars.end_times[task.task_id]
                for task in tasks
            ]

            max_earliness = LpVariable(
                f"job_{job_id}_earliness",
                lowBound=0,
                cat=LpInteger
            )

            total_earliness += max_pulp(
                model,
                earliness,
                max_earliness,
                name=f"job_{job_id}_earliness"
            )

        assert isinstance(total_earliness, ObjectiveVar)
        return total_earliness

    return export_model

@export_objective_pulp.register
def _(objective: WeightedEarliness) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
        weighted_earliness = 0
        for job_id, tasks in enumerate(objective.tasks.jobs):
            earliness = [
                objective.due_dates[job_id] - decision_vars.end_times[task.task_id]
                for task in tasks
            ]

            max_earliness = LpVariable(
                f"job_{job_id}_earliness",
                lowBound=0,
                cat=LpInteger
            )

            weighted_earliness += objective.job_weights[job_id] * max_pulp(
                model,
                earliness,
                max_earliness,
                name=f"job_{job_id}_earliness"
            )

        assert isinstance(weighted_earliness, ObjectiveVar)
        return weighted_earliness

    return export_model

@export_objective_pulp.register
def _(objective: TotalTardyJobs) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> ObjectiveVar:
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
def _(objective: WeightedTardyJobs) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    raise NotImplementedError("WeightedTardyJobs objective is not implemented for PuLP.")

@export_objective_pulp.register
def _(objective: TotalFlowTime) -> Callable[[LpProblem, PulpVariables], ObjectiveVar]:
    raise NotImplementedError("TotalFlowTime objective is not implemented for PuLP.")
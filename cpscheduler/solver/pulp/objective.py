from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, lpSum, LpVariable, LpInteger, lpDot, LpAffineExpression

from multimethod import multidispatch

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

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from .pulp_utils import max_pulp, indicator_constraint

ObjectiveVar: TypeAlias = LpVariable | LpAffineExpression
ModelExport: TypeAlias  = Callable[[LpProblem], ObjectiveVar]

@multidispatch
def export_objective_pulp(setup: Objective, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


@export_objective_pulp.register
def _(objective: ComposedObjective, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        composed_objective = LpVariable("composed_objective")

        sub_objectives = [
            export_objective_pulp(sub_objective, variables)(model)
            for sub_objective in objective.objectives
        ]

        model.addConstraint(
            composed_objective == lpDot(
                objective.coefficients,
                sub_objectives
            ),
            name="composed_objective_constraint"
        )

        return composed_objective

    return export_model

@export_objective_pulp.register
def _(objective: Makespan, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        makespan = max_pulp(
            model,
            variables.end_times,
            cat=LpInteger,
            name="makespan"
        )

        return makespan

    return export_model

@export_objective_pulp.register
def _(objective: TotalCompletionTime, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        jobs_makespan = [
            max_pulp(
                model,
                [variables.end_times[task.task_id] for task in tasks],
                cat=LpInteger,
                name=f"job_{job_id}_makespan"
            )
            for job_id, tasks in enumerate(objective.tasks.jobs)
        ]

        return lpSum(jobs_makespan)

    return export_model

@export_objective_pulp.register
def _(objective: WeightedCompletionTime, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        jobs_makespan = [
            max_pulp(
                model,
                [variables.end_times[task.task_id] for task in tasks],
                cat=LpInteger,
                name=f"job_{job_id}_makespan"
            )
            for job_id, tasks in enumerate(objective.tasks.jobs)
        ]

        return lpDot(objective.job_weights, jobs_makespan)

    return export_model

@export_objective_pulp.register
def _(objective: MaximumLateness, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        lateness = [
            end_time - due_date
            for end_time, due_date in zip(variables.end_times, objective.due_dates)
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
def _(objective: TotalTardiness, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        total_tardiness = 0
        for job_id, tasks in enumerate(objective.tasks.jobs):
            tardiness = [
                variables.end_times[task.task_id] - objective.due_dates[job_id]
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
def _(objective: WeightedTardiness, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        weighted_tardiness = 0
        for job_id, tasks in enumerate(objective.tasks.jobs):
            tardiness = [
                variables.end_times[task.task_id] - objective.due_dates[job_id]
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
def _(objective: TotalEarliness, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        total_earliness = 0
        for job_id, tasks in enumerate(objective.tasks.jobs):
            earliness = [
                objective.due_dates[job_id] - variables.end_times[task.task_id]
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
def _(objective: WeightedEarliness, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        weighted_earliness = 0
        for job_id, tasks in enumerate(objective.tasks.jobs):
            earliness = [
                objective.due_dates[job_id] - variables.end_times[task.task_id]
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
def _(objective: TotalTardyJobs, variables: PulpVariables) -> ModelExport:
    def export_model(model: LpProblem) -> ObjectiveVar:
        tardy_indicator: list[LpVariable] = []

        for tasks, due_date in zip(objective.tasks.jobs, objective.due_dates):
            max_tardiness = max_pulp(
                model,
                [
                    variables.end_times[task.task_id] - due_date for task in tasks
                ] + [0],
                cat=LpInteger,
                name=f"job_{tasks[0]}_tardiness"
            )

            tardy_indicator.append(
                    indicator_constraint(
                    model,
                    lhs=max_tardiness,
                    operator=">=",
                    rhs=1,
                    big_m=1
                )
            )

        return lpSum(tardy_indicator)

    return export_model

@export_objective_pulp.register
def _(objective: WeightedTardyJobs, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError("WeightedTardyJobs objective is not implemented for PuLP.")

@export_objective_pulp.register
def _(objective: TotalFlowTime, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError("TotalFlowTime objective is not implemented for PuLP.")

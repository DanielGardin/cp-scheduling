from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, lpSum, LpVariable, LpInteger, lpDot, LpAffineExpression

from multimethod import multidispatch

from cpscheduler.environment.tasks import Tasks
from cpscheduler.environment.data import SchedulingData
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

from .tasks import PulpVariables
from .pulp_utils import max_pulp
from ..utils import scale_to_int

ObjectiveVar: TypeAlias = LpVariable | LpAffineExpression
ModelExport: TypeAlias = Callable[[LpProblem, Tasks, SchedulingData], ObjectiveVar]


@multidispatch
def export_objective_pulp(
    objective: Objective, variables: PulpVariables
) -> ModelExport:
    raise NotImplementedError(
        f"Objective {objective} for variable {variables} not implemented for PuLP."
    )


@export_objective_pulp.register
def _(objective: ComposedObjective, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        composed_objective = LpVariable("composed_objective")

        sub_objectives = [
            export_objective_pulp(sub_objective, variables)(model, tasks, data)
            for sub_objective in objective.objectives
        ]

        coefficients = (
            scale_to_int(objective.coefficients)
            if variables.integral
            else objective.coefficients
        )

        model.addConstraint(
            composed_objective == lpDot(coefficients, sub_objectives),
            name="composed_objective_constraint",
        )

        return composed_objective

    return export_model


@export_objective_pulp.register
def _(objective: Makespan, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        makespan = max_pulp(model, variables.end_times, cat=LpInteger, name="makespan")

        return makespan

    return export_model


@export_objective_pulp.register
def _(objective: TotalCompletionTime, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        jobs_makespan = [
            max_pulp(
                model,
                [variables.end_times[task.task_id] for task in tasks],
                cat=LpInteger,
                name=f"job_{job_id}_makespan",
            )
            for job_id, tasks in enumerate(tasks.jobs)
        ]

        return lpSum(jobs_makespan)

    return export_model


@export_objective_pulp.register
def _(objective: WeightedCompletionTime, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        jobs_makespan = [
            max_pulp(
                model,
                [variables.end_times[task.task_id] for task in tasks],
                cat=LpInteger,
                name=f"job_{job_id}_makespan",
            )
            for job_id, tasks in enumerate(tasks.jobs)
        ]

        weights = (
            scale_to_int(objective.job_weights)[0]
            if variables.integral
            else objective.job_weights
        )

        return lpDot(weights, jobs_makespan)

    return export_model


@export_objective_pulp.register
def _(objective: MaximumLateness, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        lateness = [
            end_time - due_date
            for end_time, due_date in zip(variables.end_times, objective.due_dates)
        ]

        maximum_lateness = max_pulp(
            model, lateness, cat=LpInteger, name="maximum_lateness"
        )

        return maximum_lateness

    return export_model


@export_objective_pulp.register
def _(objective: TotalTardiness, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        total_tardiness = 0
        for job_id, job_tasks in enumerate(tasks.jobs):
            tardiness = [
                variables.end_times[task.task_id] - objective.due_dates[job_id]
                for task in job_tasks
            ]

            max_tardiness = LpVariable(
                f"job_{job_id}_tardiness", lowBound=0, cat=LpInteger
            )

            total_tardiness += max_pulp(
                model, tardiness, max_tardiness, name=f"job_{job_id}_tardiness"
            )

        assert isinstance(total_tardiness, ObjectiveVar)
        return total_tardiness

    return export_model


@export_objective_pulp.register
def _(objective: WeightedTardiness, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        weights = (
            scale_to_int(objective.job_weights)[0]
            if variables.integral
            else objective.job_weights
        )

        weighted_tardiness = LpAffineExpression()
        for job_id, job_tasks in enumerate(tasks.jobs):
            tardiness = [
                variables.end_times[task.task_id] - objective.due_dates[job_id]
                for task in job_tasks
            ]

            max_tardiness = LpVariable(
                f"job_{job_id}_tardiness", lowBound=0, cat=LpInteger
            )

            weighted_tardiness += weights[job_id] * max_pulp(
                model, tardiness, max_tardiness, name=f"job_{job_id}_tardiness"
            )

        return weighted_tardiness

    return export_model


@export_objective_pulp.register
def _(objective: TotalEarliness, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        total_earliness = LpAffineExpression()
        for job_id, job_tasks in enumerate(tasks.jobs):
            earliness = [
                objective.due_dates[job_id] - variables.end_times[task.task_id]
                for task in job_tasks
            ]

            max_earliness = LpVariable(
                f"job_{job_id}_earliness", lowBound=0, cat=LpInteger
            )

            total_earliness += max_pulp(
                model, earliness, max_earliness, name=f"job_{job_id}_earliness"
            )

        return total_earliness

    return export_model


@export_objective_pulp.register
def _(objective: WeightedEarliness, variables: PulpVariables) -> ModelExport:
    def export_model(
        model: LpProblem, tasks: Tasks, data: SchedulingData
    ) -> ObjectiveVar:
        weights = (
            scale_to_int(objective.job_weights)[0]
            if variables.integral
            else objective.job_weights
        )

        weighted_earliness = LpAffineExpression()
        for job_id, job_tasks in enumerate(tasks.jobs):
            earliness = [
                objective.due_dates[job_id] - variables.end_times[task.task_id]
                for task in job_tasks
            ]

            max_earliness = LpVariable(
                f"job_{job_id}_earliness", lowBound=0, cat=LpInteger
            )

            weighted_earliness += weights[job_id] * max_pulp(
                model, earliness, max_earliness, name=f"job_{job_id}_earliness"
            )

        return weighted_earliness

    return export_model


@export_objective_pulp.register
def _(objective: TotalTardyJobs, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError("TotalTardyJobs objective is not implemented for PuLP.")


@export_objective_pulp.register
def _(objective: WeightedTardyJobs, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError(
        "WeightedTardyJobs objective is not implemented for PuLP."
    )


@export_objective_pulp.register
def _(objective: TotalFlowTime, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError("TotalFlowTime objective is not implemented for PuLP.")

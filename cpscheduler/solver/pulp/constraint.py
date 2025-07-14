from typing import TypeAlias
from collections.abc import Callable

from itertools import combinations

from pulp import LpProblem, lpSum

from multimethod import multidispatch

from cpscheduler.environment.tasks import Tasks
from cpscheduler.environment.data import SchedulingData
from cpscheduler.environment.constraints import (
    Constraint,
    PrecedenceConstraint,
    ConstantProcessingTime,
    DisjunctiveConstraint,
    ReleaseDateConstraint,
    DeadlineConstraint,
    ResourceConstraint,
    MachineConstraint,
    SetupConstraint,
)

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from .pulp_utils import implication_pulp, pulp_add_constraint, PULP_EXPRESSION

ModelExport: TypeAlias = Callable[[LpProblem, Tasks, SchedulingData], None]


@multidispatch
def export_constraint_pulp(
    constraint: Constraint, variables: PulpVariables
) -> ModelExport:
    raise NotImplementedError(
        f"Constraint {constraint} for variable {variables} not implemented for PuLP."
    )


@export_constraint_pulp.register
def _(
    constraint: PrecedenceConstraint, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for task_id, children in constraint.precedence.items():
            for child_id in children:
                pulp_add_constraint(
                    model,
                    (
                        variables.end_times[task_id] == variables.start_times[child_id]
                        if constraint.no_wait
                        else variables.end_times[task_id]
                        <= variables.start_times[child_id]
                    ),
                    f"precedence_{task_id}_{child_id}",
                )

                pulp_add_constraint(
                    model,
                    variables.get_order(task_id, child_id) >= 1,
                    f"order_{task_id}_{child_id}",
                )

    return export_model


@export_constraint_pulp.register
def _(
    constraint: DisjunctiveConstraint, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for _, group_tasks in constraint.disjunctive_groups.items():
            for i, j in combinations(group_tasks, 2):
                implication_pulp(
                    model,
                    antecedent=variables.get_order(i, j),
                    consequent=(
                        variables.end_times[i],
                        "<=",
                        variables.start_times[j],
                    ),
                    big_m=int(tasks[i].get_end_ub() - tasks[j].get_start_lb()),
                    name=f"disjunctive_{i}_{j}_order",
                )

                implication_pulp(
                    model,
                    antecedent=variables.get_order(j, i),
                    consequent=(
                        variables.end_times[j],
                        "<=",
                        variables.start_times[i],
                    ),
                    big_m=int(tasks[j].get_end_ub() - tasks[i].get_start_lb()),
                    name=f"disjunctive_{j}_{i}_order",
                )

    return export_model


@export_constraint_pulp.register
def _(
    constraint: ReleaseDateConstraint | DeadlineConstraint | ConstantProcessingTime,
    variables: PulpSchedulingVariables,
) -> ModelExport:
    return (
        lambda model, tasks, data: None
    )  # No specific export needed for these constraints


@export_constraint_pulp.register
def _(
    constraint: ResourceConstraint, variables: PulpSchedulingVariables
) -> ModelExport:
    raise NotImplementedError(
        "Resource constraints are not available in PuLP at the moment."
    )


@export_constraint_pulp.register
def _(constraint: MachineConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        machine_constraint: list[list[int]] = [[] for _ in range(data.n_machines)]

        for task in tasks:
            for machine_id in task.machines:
                machine_constraint[machine_id].append(task.task_id)

        for machine_id, machine_tasks in enumerate(machine_constraint):
            for i, j in combinations(machine_tasks, 2):
                implication_pulp(
                    model,
                    antecedent=(
                        variables.get_order(i, j),
                        variables.assignments[i][machine_id],
                        variables.assignments[j][machine_id],
                    ),
                    consequent=(
                        variables.end_times[i],
                        "<=",
                        variables.start_times[j],
                    ),
                    big_m=int(tasks[i].get_end_ub() - tasks[j].get_start_lb()),
                    name=f"machine_{machine_id}_disjunctive_{i}_{j}_order",
                )

                implication_pulp(
                    model,
                    antecedent=(
                        variables.get_order(j, i),
                        variables.assignments[i][machine_id],
                        variables.assignments[j][machine_id],
                    ),
                    consequent=(
                        variables.end_times[j],
                        "<=",
                        variables.start_times[i],
                    ),
                    big_m=int(tasks[j].get_end_ub() - tasks[i].get_start_lb()),
                    name=f"machine_{machine_id}_disjunctive_{j}_{i}_order",
                )

    return export_model


@export_constraint_pulp.register
def _(constraint: SetupConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for task_id, setup_times in constraint.setup_times.items():
            for child_id, setup_time in setup_times.items():

                implication_pulp(
                    model,
                    antecedent=variables.get_order(task_id, child_id),
                    consequent=(
                        variables.end_times[task_id] + setup_time,
                        "<=",
                        variables.start_times[child_id],
                    ),
                    big_m=int(
                        tasks[task_id].get_end_ub()
                        + setup_time
                        - tasks[child_id].get_start_lb()
                    ),
                    name=f"setup_{task_id}_{child_id}_order",
                )

    return export_model

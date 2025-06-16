from typing import TypeAlias
from collections.abc import Callable, Iterable

from pulp import LpProblem

from functools import singledispatch

from cpscheduler.environment.constraints import (
    Constraint,
    PrecedenceConstraint,
    DisjunctiveConstraint,
    ReleaseDateConstraint,
    DeadlineConstraint,
    ResourceConstraint,
    MachineConstraint,
    SetupConstraint,
)

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from .utils import indicator_constraint

ModelExport: TypeAlias = Callable[[LpProblem], None]

@singledispatch
def export_constraint_pulp(setup: Constraint, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")

@export_constraint_pulp.register
def _(constraint: PrecedenceConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        for task_id, children in constraint.precedence.items():
            for child_id in children:
                model.addConstraint(
                    variables.end_times[task_id] == variables.start_times[child_id]
                    if constraint.no_wait else
                    variables.end_times[task_id] <= variables.start_times[child_id],
                    f"precedence_{task_id}_{child_id}"
                )

                if task_id < child_id:
                    model.addConstraint(
                        variables.orders[(task_id, child_id)] == 1
                    )
                
                else:
                    model.addConstraint(
                        variables.orders[(task_id, child_id)] == 0
                    )

    return export_model

# @export_constraint_pulp.register
# def _(constraint: PrecedenceConstraint, variables: PulpTimetable) -> ModelExport:
#     def export_model(model: LpProblem) -> None:
#         for task_id, children in constraint.precedence.items():
#             for child_id in children:



@export_constraint_pulp.register
def _(constraint: DisjunctiveConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        for _, tasks in constraint.disjunctive_groups.items():
            n_tasks = len(tasks)

            for i in range(n_tasks):
                for j in range(i):
                    order = (min(tasks[i], tasks[j]), max(tasks[i], tasks[j]))

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[tasks[i]],
                        operator   = "<=",
                        rhs        = variables.start_times[tasks[j]],
                        indicators = variables.orders[order],
                        big_m      = constraint.tasks[tasks[i]].get_end_ub()
                    )

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[tasks[j]],
                        operator   = "<=",
                        rhs        = variables.start_times[tasks[i]],
                        indicators = 1 - variables.orders[order],
                        big_m      = constraint.tasks[tasks[j]].get_end_ub()
                    )

    return export_model

@export_constraint_pulp.register
def _(constraint: ReleaseDateConstraint | DeadlineConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    return lambda model: None  # No specific export needed for these constraints

@export_constraint_pulp.register
def _(constraint: ResourceConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    raise NotImplementedError("Resource constraints are not available in PuLP at the moment.")

@export_constraint_pulp.register
def _(constraint: MachineConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        tasks_per_machine: dict[int, list[int]] = {}

        if constraint.complete:
            tasks_per_machine = {
                machine: list(range(constraint.tasks.n_tasks))
                for machine in range(constraint.tasks.n_machines)
            }

        for task_id, machines in enumerate(constraint.machine_constraint):
            for machine in machines:
                if machine not in tasks_per_machine:
                    tasks_per_machine[machine] = []

                tasks_per_machine[machine].append(task_id)

        for machine, tasks in tasks_per_machine.items():
            for i in range(len(tasks)):
                for j in range(i):
                    


                    order = (min(tasks[i], tasks[j]), max(tasks[i], tasks[j]))

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[tasks[i]],
                        operator   = "<=",
                        rhs        = variables.start_times[tasks[j]],
                        indicators = (
                            variables.orders[order],
                            variables.assignments[tasks[i]][machine],
                            variables.assignments[tasks[j]][machine]
                        ),
                        big_m      = constraint.tasks[tasks[i]].get_end_ub()
                    )

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[tasks[j]],
                        operator   = "<=",
                        rhs        = variables.start_times[tasks[i]],
                        indicators = (
                            1 - variables.orders[order],
                            variables.assignments[tasks[i]][machine],
                            variables.assignments[tasks[j]][machine]
                        ),
                        big_m      = constraint.tasks[tasks[j]].get_end_ub()
                    )

    return export_model

@export_constraint_pulp.register
def _(constraint: SetupConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
            for task_id, setup_times in constraint.setup_times.items():
                for child_id, setup_time in setup_times.items():
                    order = (min(task_id, child_id), max(task_id, child_id))

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[task_id] + setup_time,
                        operator   = "<=",
                        rhs        = variables.start_times[child_id],
                        indicators = variables.orders[order],
                        big_m      = constraint.tasks[task_id].get_end_ub()
                    )
        
    return export_model

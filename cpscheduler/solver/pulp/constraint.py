from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, LpVariable, lpSum

from multimethod import multidispatch

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
from .pulp_utils import indicator_constraint

ModelExport: TypeAlias = Callable[[LpProblem], None]

@multidispatch
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
                        variables.orders[(task_id, child_id)] == 1,
                        f"order_{task_id}_{child_id}"
                    )

                else:
                    model.addConstraint(
                        variables.orders[(task_id, child_id)] == 0,
                        f"order_{child_id}_{task_id}"
                    )

    return export_model

@export_constraint_pulp.register
def _(constraint: DisjunctiveConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        for _, tasks in constraint.disjunctive_groups.items():
            n_tasks = len(tasks)

            for i in range(n_tasks):
                for j in range(i):
                    task_i = min(tasks[i], tasks[j])
                    task_j = max(tasks[i], tasks[j])

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[task_i],
                        operator   = "<=",
                        rhs        = variables.start_times[task_j],
                        indicators = variables.orders[task_i, task_j],
                        big_m      = (
                            constraint.tasks[task_i].get_end_ub() -
                            constraint.tasks[task_j].get_start_lb()
                        ),
                        name=f"disjunctive_{task_i}_{task_j}_order"
                    )

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[task_j],
                        operator   = "<=",
                        rhs        = variables.start_times[task_i],
                        indicators = 1 - variables.orders[task_i, task_j],
                        big_m      = (
                            constraint.tasks[task_j].get_end_ub() -
                            constraint.tasks[task_i].get_start_lb()
                        ),
                        name=f"disjunctive_{task_j}_{task_i}_order"
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
                    task_i = min(tasks[i], tasks[j])
                    task_j = max(tasks[i], tasks[j])

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[task_i],
                        operator   = "<=",
                        rhs        = variables.start_times[task_j],
                        indicators = (
                            variables.orders[task_i, task_j],
                            variables.assignments[task_i][machine],
                            variables.assignments[task_j][machine]
                        ),
                        big_m      = (
                            constraint.tasks[task_i].get_end_ub() -
                            constraint.tasks[task_j].get_start_lb()
                        ),
                        name=f"machine_{machine}_disjunctive_{task_i}_{task_j}_order"
                    )

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[task_j],
                        operator   = "<=",
                        rhs        = variables.start_times[task_i],
                        indicators = (
                            1 - variables.orders[task_i, task_j],
                            variables.assignments[task_i][machine],
                            variables.assignments[task_j][machine]
                        ),
                        big_m      = (
                            constraint.tasks[task_j].get_end_ub() -
                            constraint.tasks[task_i].get_start_lb()
                        ),
                        name=f"machine_{machine}_disjunctive_{task_j}_{task_i}_order"
                    )

    return export_model

@export_constraint_pulp.register
def _(constraint: SetupConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
            for task_id, setup_times in constraint.setup_times.items():
                for child_id, setup_time in setup_times.items():

                    indicator_constraint(
                        model,
                        lhs        = variables.end_times[task_id] + setup_time,
                        operator   = "<=",
                        rhs        = variables.start_times[child_id],
                        indicators = (
                            variables.orders[task_id, child_id] if task_id < child_id else
                            1 - variables.orders[child_id, task_id],
                        ),
                        big_m      = (
                            constraint.tasks[task_id].get_end_ub() + setup_time -
                            constraint.tasks[child_id].get_start_lb()
                        ),
                        name=f"setup_{task_id}_{child_id}_order"
                    )

    return export_model

# Timetable implementations

# @export_constraint_pulp.register
# def _(constraint: PrecedenceConstraint, variables: PulpTimetable) -> ModelExport:
#     def export_model(model: LpProblem) -> None:
#         for task_id, children in constraint.precedence.items():
#             for child_id in children:
#                 for child_start_time in range(variables.T):
#                     ...



@export_constraint_pulp.register
def _(constraint: MachineConstraint, variables: PulpTimetable) -> ModelExport:
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
            for time in range(variables.T):
                disjunction_group: list[LpVariable] = []

                for task_id in tasks:
                    task = constraint.tasks[task_id]
                    start_lb = task.get_start_lb()

                    if time < start_lb or time >= task.get_start_ub():
                        continue

                    start_time = time - task.processing_times[machine] + 1
                    start_time = max(start_time, start_lb)

                    disjunction_group.extend(
                        variables.start_times[task_id][machine][start_time:time+1]
                    )

                model.addConstraint(
                    lpSum(disjunction_group) <= 1,
                    f"machine_{machine}_timetable_{time}"
                )

    return export_model
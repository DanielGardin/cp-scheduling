from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, LpVariable, lpSum

from multimethod import multidispatch

from cpscheduler.environment._common import MACHINE_ID
from cpscheduler.environment.tasks import Tasks
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

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable, get_order
from .pulp_utils import implication_pulp

ModelExport: TypeAlias = Callable[[LpProblem, Tasks], None]


@multidispatch
def export_constraint_pulp(setup: Constraint, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


@export_constraint_pulp.register
def _(
    constraint: PrecedenceConstraint, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        for task_id, children in constraint.precedence.items():
            for child_id in children:
                model.addConstraint(
                    (
                        variables.end_times[task_id] == variables.start_times[child_id]
                        if constraint.no_wait
                        else variables.end_times[task_id]
                        <= variables.start_times[child_id]
                    ),
                    f"precedence_{task_id}_{child_id}",
                )

                if task_id < child_id:
                    model.addConstraint(
                        variables.orders[(task_id, child_id)] == 1,
                        f"order_{task_id}_{child_id}",
                    )

                else:
                    model.addConstraint(
                        variables.orders[(task_id, child_id)] == 0,
                        f"order_{child_id}_{task_id}",
                    )

    return export_model


@export_constraint_pulp.register
def _(
    constraint: DisjunctiveConstraint, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        for _, group_tasks in constraint.disjunctive_groups.items():
            n_group_tasks = len(group_tasks)

            for i in range(n_group_tasks):
                for j in range(i):
                    order, _ = get_order(group_tasks[i], group_tasks[j])
                    task_i, task_j = order

                    implication_pulp(
                        model,
                        antecedent=variables.orders[order],
                        consequent=(
                            variables.end_times[task_i],
                            "<=",
                            variables.start_times[task_j],
                        ),
                        big_m=int(
                            tasks[task_i].get_end_ub() - tasks[task_j].get_start_lb()
                        ),
                        name=f"disjunctive_{task_i}_{task_j}_order",
                    )

                    implication_pulp(
                        model,
                        antecedent=1 - variables.orders[order],
                        consequent=(
                            variables.end_times[task_j],
                            "<=",
                            variables.start_times[task_i],
                        ),
                        big_m=int(
                            tasks[task_j].get_end_ub() - tasks[task_i].get_start_lb()
                        ),
                        name=f"disjunctive_{task_j}_{task_i}_order",
                    )

    return export_model


@export_constraint_pulp.register
def _(
    constraint: ReleaseDateConstraint | DeadlineConstraint,
    variables: PulpSchedulingVariables,
) -> ModelExport:
    return lambda model, tasks: None  # No specific export needed for these constraints


@export_constraint_pulp.register
def _(
    constraint: ResourceConstraint, variables: PulpSchedulingVariables
) -> ModelExport:
    raise NotImplementedError(
        "Resource constraints are not available in PuLP at the moment."
    )


@export_constraint_pulp.register
def _(constraint: MachineConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        tasks_per_machine: dict[MACHINE_ID, list[int]] = {}

        if constraint.complete:
            tasks_per_machine = {
                machine: list(range(tasks.n_tasks))
                for machine in range(tasks.n_machines)
            }

        for task_id, machines in enumerate(constraint.machine_constraint):
            for machine in machines:
                if machine not in tasks_per_machine:
                    tasks_per_machine[machine] = []

                tasks_per_machine[machine].append(task_id)

        for machine, machine_tasks in tasks_per_machine.items():
            for i in range(len(tasks)):
                for j in range(i):
                    order, _ = get_order(machine_tasks[i], machine_tasks[j])
                    task_i, task_j = order

                    implication_pulp(
                        model,
                        antecedent=(
                            variables.orders[task_i, task_j],
                            variables.assignments[task_i][machine],
                            variables.assignments[task_j][machine],
                        ),
                        consequent=(
                            variables.end_times[task_i],
                            "<=",
                            variables.start_times[task_j],
                        ),
                        big_m=int(
                            tasks[task_i].get_end_ub() - tasks[task_j].get_start_lb()
                        ),
                        name=f"machine_{machine}_disjunctive_{task_i}_{task_j}_order",
                    )

                    implication_pulp(
                        model,
                        antecedent=(
                            1 - variables.orders[task_i, task_j],
                            variables.assignments[task_i][machine],
                            variables.assignments[task_j][machine],
                        ),
                        consequent=(
                            variables.end_times[task_j],
                            "<=",
                            variables.start_times[task_i],
                        ),
                        big_m=int(
                            tasks[task_j].get_end_ub() - tasks[task_i].get_start_lb()
                        ),
                        name=f"machine_{machine}_disjunctive_{task_j}_{task_i}_order",
                    )

    return export_model


@export_constraint_pulp.register
def _(constraint: SetupConstraint, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        for task_id, setup_times in constraint.setup_times.items():
            for child_id, setup_time in setup_times.items():

                implication_pulp(
                    model,
                    antecedent=(
                        (
                            variables.orders[task_id, child_id]
                            if task_id < child_id
                            else 1 - variables.orders[child_id, task_id]
                        ),
                    ),
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


# Timetable implementations

# @export_constraint_pulp.register
# def _(constraint: PrecedenceConstraint, variables: PulpTimetable) -> ModelExport:
#     def export_model(model: LpProblem, tasks: Tasks) -> None:
#         for task_id, children in constraint.precedence.items():
#             for child_id in children:
#                 for child_start_time in range(variables.T):
#                     ...


@export_constraint_pulp.register
def _(constraint: MachineConstraint, variables: PulpTimetable) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        tasks_per_machine: dict[int, list[int]] = {}

        if constraint.complete:
            tasks_per_machine = {
                machine: list(range(tasks.n_tasks))
                for machine in range(tasks.n_machines)
            }

        for task_id, machines in enumerate(constraint.machine_constraint):
            for machine in machines:
                if machine not in tasks_per_machine:
                    tasks_per_machine[machine] = []

                tasks_per_machine[machine].append(task_id)

        for machine, machine_tasks in tasks_per_machine.items():
            for time in range(variables.T):
                disjunction_group: list[LpVariable] = []

                for task_id in machine_tasks:
                    task = tasks[task_id]
                    start_lb = task.get_start_lb()

                    if time < start_lb or time >= task.get_start_ub():
                        continue

                    start_time = time - task.processing_times[machine] + 1
                    start_time = max(start_time, start_lb)

                    disjunction_group.extend(
                        variables.start_times[task_id][machine][start_time : time + 1]
                    )

                model.addConstraint(
                    lpSum(disjunction_group) <= 1, f"machine_{machine}_timetable_{time}"
                )

    return export_model

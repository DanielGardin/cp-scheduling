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

from .tasks import PulpVariables
from .utils import indicator_var

@singledispatch
def export_constraint_pulp(setup: Constraint,) -> Callable[[LpProblem, PulpVariables], None]:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")

@export_constraint_pulp.register
def _(constraint: PrecedenceConstraint) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
        for task_id, children in constraint.precedence.items():
            for child in children:
                model.addConstraint(
                    decision_vars.end_times[task_id] == decision_vars.start_times[child]
                    if constraint.no_wait else
                    decision_vars.end_times[task_id] <= decision_vars.start_times[child],
                    f"precedence_{task_id}_{child}"
                )

    return export_model

@export_constraint_pulp.register
def _(constraint: DisjunctiveConstraint) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
        for _, tasks in constraint.disjunctive_groups.items():
            n_tasks = len(tasks)

            for i in range(n_tasks):
                for j in range(i):
                    indicator_var(
                        model,
                        lhs       = decision_vars.end_times[tasks[i]],
                        operator  = "<=",
                        rhs       = decision_vars.start_times[tasks[j]],
                        indicator = decision_vars.orders[(tasks[i], tasks[j])],
                        big_m     = constraint.tasks[tasks[i]].get_end_ub()
                    )

                    indicator_var(
                        model,
                        lhs       = decision_vars.end_times[tasks[j]],
                        operator  = "<=",
                        rhs       = decision_vars.start_times[tasks[i]],
                        indicator = 1 - decision_vars.orders[(tasks[i], tasks[j])],
                        big_m     = constraint.tasks[tasks[j]].get_end_ub()
                    )

    return export_model

@export_constraint_pulp.register
def _(constraint: ReleaseDateConstraint | DeadlineConstraint) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
        return

    return export_model

@export_constraint_pulp.register
def _(constraint: ResourceConstraint) -> Callable[[LpProblem, PulpVariables], None]:
    raise NotImplementedError("Resource constraints are not available in PuLP at the moment.")

@export_constraint_pulp.register
def _(constraint: MachineConstraint) -> Callable[[LpProblem, PulpVariables], None]:
    raise NotImplementedError("Machine constraints are not available in PuLP at the moment.")

@export_constraint_pulp.register
def _(constraint: SetupConstraint) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
            for task_id, setup_times in constraint.setup_times.items():
                for child_id, setup_time in setup_times.items():
                    indicator_var(
                        model,
                        lhs       = decision_vars.end_times[task_id] + setup_time,
                        operator  = "<=",
                        rhs       = decision_vars.start_times[child_id],
                        indicator = decision_vars.orders[(task_id, child_id)],
                        big_m     = constraint.tasks[task_id].get_end_ub()
                    )
        
    return export_model
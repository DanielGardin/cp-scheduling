from collections.abc import Iterable

from pulp import LpProblem, LpAffineExpression, lpSum

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable, get_order
from .pulp_utils import implication_pulp
from itertools import repeat, combinations

from cpscheduler.environment.env import SchedulingEnv

import cpscheduler.environment.constraints as constraints
import cpscheduler.environment.objectives as objectives
import cpscheduler.environment.schedule_setup as setups


def is_partition_problem(env: SchedulingEnv) -> bool:
    "Check if the environment has no additional constraints (beta entries)."
    constraint = constraints.Constraint()
    for constraint in env.constraints.values():
        if isinstance(constraint, constraints.MachineConstraint):
            return constraint.is_complete()

        return False

    return True


def employ_symmetry_breaking_pulp(
    env: SchedulingEnv,
    model: LpProblem,
    variables: PulpVariables,
) -> None:
    if not isinstance(variables, PulpSchedulingVariables):
        return

    if is_partition_problem(env):
        partition_symmetry_breaking(env, model, variables)
        return


def partition_symmetry_breaking(
    env: SchedulingEnv,
    model: LpProblem,
    variables: PulpSchedulingVariables,
) -> None:
    """
    In partition problems, the start and end times are not essential part of the solution.
    To break the symmetry of different start and end times being equivalent, we can
    fix them (or their order) using dominance results with pre-calculated start and end times,
    or ordering, based uniquely on their assignments.
    """
    # When machines are exchangeable (identical machines),
    # we can break symmetry by ordering the machines
    if isinstance(env.setup, setups.IdenticalParallelMachineSetup):
        machine_ordering_symmetry_breaking(env, model, variables)

    # The objective appoint the dominance rules among tasks, for example,
    # Completion time tasks are dominated by the tasks with smaller processing times.
    # In Makespan problems the tasks are fully exchangeable.
    if isinstance(
        env.objective,
        (objectives.WeightedCompletionTime, objectives.TotalCompletionTime),
    ):
        smiths_rules_symmetry_breaking(env, model, variables)

    elif isinstance(env.objective, objectives.Makespan):
        job_ordering_symmetry_breaking(env, model, variables)

    elif isinstance(
        env.objective,
        (objectives.WeightedTardiness, objectives.TotalTardiness),
    ):
        ...


def machine_ordering_symmetry_breaking(
    env: SchedulingEnv,
    model: LpProblem,
    decision_vars: PulpSchedulingVariables,
) -> None:
    "When machines are exchangeable, break symmetry by ordering by load"
    tasks = env.tasks
    data = env.data
    n_machines = data.n_machines

    processing_times: list[LpAffineExpression] = [
        lpSum(
            [
                task.processing_times[machine_id]
                * decision_vars.assignments[task_id][machine_id]
                for task_id, task in enumerate(tasks)
                if machine_id in task.processing_times
            ]
        )
        for machine_id in range(n_machines)
    ]

    for machine_id in range(n_machines - 1):
        model.addConstraint(
            processing_times[machine_id] >= processing_times[machine_id + 1],
            name=f"SB_machine_{machine_id}_order",
        )


def job_ordering_symmetry_breaking(
    env: SchedulingEnv,
    model: LpProblem,
    decision_vars: PulpSchedulingVariables,
) -> None:
    "When jobs inside machines are exchangeable, break symmetry by ordering by lexicographic order"
    tasks = env.tasks

    machine_constraint = next(
        iter(
            constraint
            for constraint in env.constraints.values()
            if isinstance(constraint, constraints.MachineConstraint)
        )
    )

    for machine_id, machine_tasks in enumerate(machine_constraint.machine_constraint):
        for i, j in combinations(machine_tasks, 2):
            implication_pulp(
                model,
                antecedent=(
                    decision_vars.assignments[i][machine_id],
                    decision_vars.assignments[j][machine_id],
                ),
                consequent=(decision_vars.get_order(i, j), "==", 1),
                big_m=1,
                name=f"SB_order_{i}_{j}_machine_{machine_id}",
            )

            implication_pulp(
                model,
                antecedent=(
                    decision_vars.assignments[i][machine_id],
                    decision_vars.assignments[j][machine_id],
                ),
                consequent=(
                    decision_vars.end_times[i],
                    "<=",
                    decision_vars.start_times[j],
                ),
                big_m=int(tasks[i].get_end_ub() - tasks[j].get_start_lb()),
                name=f"SB_job_{i}_{j}_machine_{machine_id}",
            )


def smiths_rules_symmetry_breaking(
    env: SchedulingEnv, model: LpProblem, decision_vars: PulpSchedulingVariables
) -> None:
    """
    In partition problems, break symmetry by ordering by Smith's rules.
    The smith's rule states that there is a optimal solution such that
    the task
    """
    tasks = env.tasks
    data = env.data

    weights: Iterable[float]
    if isinstance(env.objective, objectives.WeightedCompletionTime):
        weights = env.objective.job_weights

    elif isinstance(env.objective, objectives.TotalCompletionTime):
        weights = repeat(1.0, len(tasks))

    else:
        raise ValueError(
            "Smith's rules symmetry breaking is only applicable"
            "for WeightedCompletionTime or TotalCompletionTime objectives."
        )

    priorities: list[tuple[float, int]]
    for machine_id in range(data.n_machines):
        priorities = sorted(
            [
                (-weight / int(task.processing_times[machine_id]), task.task_id)
                for task, weight in zip(tasks, weights)
                if machine_id in task.processing_times
            ]
        )

        for i, (_, task_id) in enumerate(priorities):
            S_j = lpSum(
                [
                    tasks[prev_task].processing_times[machine_id]
                    * decision_vars.assignments[prev_task][machine_id]
                    for _, prev_task in priorities[:i]
                ]
            )

            implication_pulp(
                model,
                antecedent=decision_vars.assignments[task_id][machine_id],
                consequent=(decision_vars.start_times[task_id], "==", S_j),
                big_m=int(
                    tasks[task_id].get_start_ub() - tasks[task_id].get_start_lb()
                ),
                name=f"SB_smiths_{task_id}_machine_{machine_id}",
            )

            for _, prev_task in priorities[:i]:
                order, value = get_order(task_id, prev_task)

                implication_pulp(
                    model,
                    antecedent=(
                        decision_vars.assignments[task_id][machine_id],
                        decision_vars.assignments[prev_task][machine_id],
                    ),
                    consequent=(decision_vars.orders[order], "==", value),
                    big_m=1,
                    name=f"SB_smiths_order_{order[0]}_{order[1]}_machine_{machine_id}",
                )

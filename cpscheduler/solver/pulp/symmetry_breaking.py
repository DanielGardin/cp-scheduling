from itertools import combinations
from functools import partial

from pulp import LpProblem, LpAffineExpression, lpSum

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from .pulp_utils import implication_pulp, and_pulp, pulp_add_constraint

from cpscheduler.environment.env import SchedulingEnv

import cpscheduler.environment.constraints as constraints
import cpscheduler.environment.objectives as objectives
import cpscheduler.environment.schedule_setup as setups


def get_machine_constraint(env: SchedulingEnv) -> constraints.MachineConstraint | None:
    for constraint in env.constraints.values():
        if isinstance(constraint, constraints.MachineConstraint):
            return constraint
    return None


def is_partition_problem(env: SchedulingEnv) -> bool:
    "Check if the environment has no additional constraints (beta entries)."
    constraint = get_machine_constraint(env)

    if constraint is None:
        return False

    return constraint.is_complete(env.tasks)


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
        tardiness_dominance_symmetry_breaking(env, model, variables)


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
                task._remaining_times[machine_id]
                * decision_vars.assignments[task_id][machine_id]
                for task_id, task in enumerate(tasks)
                if machine_id in task._remaining_times
            ]
        )
        for machine_id in range(n_machines)
    ]

    for machine_id in range(n_machines - 1):
        pulp_add_constraint(
            model,
            processing_times[machine_id] <= processing_times[machine_id + 1],
            name=f"SB_machine_{machine_id}_load",
        )


def job_ordering_symmetry_breaking(
    env: SchedulingEnv,
    model: LpProblem,
    decision_vars: PulpSchedulingVariables,
) -> None:
    "When jobs inside machines are exchangeable, break symmetry by ordering by lexicographic order"
    tasks = env.tasks

    machine_constraint: list[list[int]] = [[] for _ in range(env.data.n_machines)]

    for task in tasks:
        for machine_id in task.machines:
            machine_constraint[machine_id].append(task.task_id)

    for machine_id, machine_tasks in enumerate(machine_constraint):
        for i, j in combinations(machine_tasks, 2):
            implication_pulp(
                model,
                antecedent=(
                    decision_vars.assignments[i][machine_id],
                    decision_vars.assignments[j][machine_id],
                ),
                consequent=(decision_vars.get_order(i, j), ">=", 1),
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

    if isinstance(env.objective, objectives.WeightedCompletionTime):
        weighted = True
        weights: list[float] = env.data.get_job_level_data("weight")

    elif isinstance(env.objective, objectives.TotalCompletionTime):
        weighted = False
        weights = []

    else:
        raise ValueError(
            "Smith's rules symmetry breaking is only applicable"
            "for WeightedCompletionTime or TotalCompletionTime objectives."
        )

    machine_constraint: list[list[int]] = [[] for _ in range(env.data.n_machines)]

    for task in tasks:
        for machine_id in task.machines:
            machine_constraint[machine_id].append(task.task_id)

    def smith_rule(task_id: int, machine_id: int) -> float:
        task = tasks[task_id]
        if weighted:
            return -weights[task.job_id] / int(task._remaining_times[machine_id])

        return int(task._remaining_times[machine_id])

    for machine_id, machine_tasks in enumerate(machine_constraint):
        priorities = sorted(
            machine_tasks, key=partial(smith_rule, machine_id=machine_id)
        )

        for idx, task_id in enumerate(priorities):
            S_j = lpSum(
                tasks[prev_task_id]._remaining_times[machine_id]
                * decision_vars.assignments[prev_task_id][machine_id]
                for prev_task_id in priorities[:idx]
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

            for prev_task_id in priorities[:idx]:
                implication_pulp(
                    model,
                    antecedent=(
                        decision_vars.assignments[task_id][machine_id],
                        decision_vars.assignments[prev_task_id][machine_id],
                    ),
                    consequent=(
                        decision_vars.get_order(prev_task_id, task_id),
                        ">=",
                        1,
                    ),
                    big_m=1,
                    name=f"SB_smiths_order_{prev_task_id}_{task_id}_machine_{machine_id}",
                )


def tardiness_dominance_symmetry_breaking(
    env: SchedulingEnv, model: LpProblem, decision_vars: PulpSchedulingVariables
) -> None:
    "In agreeable jobs, break symmetry by ordering by tardiness dominance"
    tasks = env.tasks

    machine_constraint: list[list[int]] = [[] for _ in range(env.data.n_machines)]

    for task in tasks:
        for machine_id in task.machines:
            machine_constraint[machine_id].append(task.task_id)

    if isinstance(env.objective, objectives.WeightedTardiness):
        weighted = True
        weights: list[float] = env.data.get_job_level_data("weight")

    elif isinstance(env.objective, objectives.TotalTardiness):
        weighted = False
        weights = []

    else:
        raise ValueError(
            "Tardiness dominance symmetry breaking is only applicable"
            "for WeightedTardiness or TotalTardiness objectives."
        )

    due_dates = env.data.get_job_level_data("due_date")

    for machine_id, machine_tasks in enumerate(machine_constraint):
        sorted_tasks = sorted(
            machine_tasks, key=lambda task_id: due_dates[tasks[task_id].job_id]
        )

        # Necessarely d_i <= d_j
        for i, j in combinations(sorted_tasks, 2):
            task_i = tasks[i]
            task_j = tasks[j]

            if task_i._remaining_times[machine_id] >= task_j._remaining_times[
                machine_id
            ] or (weighted and weights[task_i.job_id] <= weights[task_j.job_id]):
                continue

            implication_pulp(
                model,
                antecedent=(
                    decision_vars.assignments[i][machine_id],
                    decision_vars.assignments[j][machine_id],
                ),
                consequent=(
                    decision_vars.get_order(i, j),
                    ">=",
                    1,
                ),
                big_m=1,
                name=f"SB_tardiness_order_{i}_{j}_machine_{machine_id}",
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
                big_m=int(task_i.get_end_ub() - task_j.get_start_lb()),
                name=f"SB_tardiness_{i}_{j}_machine_{machine_id}",
            )

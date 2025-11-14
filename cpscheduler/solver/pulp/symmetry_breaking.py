from itertools import combinations
from functools import partial

from pulp import LpProblem, LpAffineExpression, lpSum

from cpscheduler.solver.pulp.tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from cpscheduler.solver.pulp.pulp_utils import implication_pulp, and_pulp, pulp_add_constraint

from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.tasks import Task

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

    return constraint.is_complete(env.state)


def employ_symmetry_breaking_pulp(
    env: SchedulingEnv,
    model: LpProblem,
    variables: PulpVariables,
) -> None:
    if not isinstance(variables, PulpSchedulingVariables):
        return

    start_lower_bound_symmetry_breaking(env, model, variables)

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


def start_lower_bound_symmetry_breaking(
    env: SchedulingEnv, model: LpProblem, decision_vars: PulpSchedulingVariables
) -> None:
    lower_bounds = [LpAffineExpression() for _ in range(env.state.n_tasks)]

    for (i, j), order in decision_vars.orders.items():
        task_i = env.state.tasks[i]
        task_j = env.state.tasks[j]

        machine_i = set(task_i.machines)
        machine_j = set(task_j.machines)

        for machine in machine_i.intersection(machine_j):
            allocate_forward = and_pulp(
                model,
                (
                    decision_vars.assignments[i][machine],
                    decision_vars.assignments[j][machine],
                    order,
                ),
            )

            lower_bounds[j] += task_i.remaining_times[machine] * allocate_forward

            allocate_backward = and_pulp(
                model,
                (
                    decision_vars.assignments[i][machine],
                    decision_vars.assignments[j][machine],
                    1 - order,
                ),
            )

            lower_bounds[i] += (
                task_j.remaining_times[machine] * allocate_backward
            )

    for task_id in range(env.state.n_tasks):
        pulp_add_constraint(
            model,
            lower_bounds[task_id] <= decision_vars.start_times[task_id],
            name=f"SB_start_lower_bound_{task_id}",
        )


def machine_ordering_symmetry_breaking(
    env: SchedulingEnv,
    model: LpProblem,
    decision_vars: PulpSchedulingVariables,
) -> None:
    "When machines are exchangeable, break symmetry by ordering by load"
    state = env.state
    n_machines = state.n_machines

    processing_times: list[LpAffineExpression] = [
        lpSum(
            [
                task.remaining_times[machine_id]
                * decision_vars.assignments[task_id][machine_id]
                for task_id, task in enumerate(state.tasks)
                if machine_id in task.remaining_times
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
    state = env.state

    machine_constraint: list[list[int]] = [[] for _ in range(state.n_machines)]

    for task in state.tasks:
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
                big_m=int(state.tasks[i].get_end_ub() - state.tasks[j].get_start_lb()),
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
    state = env.state

    if isinstance(env.objective, objectives.WeightedCompletionTime):
        weighted = True
        weights: list[float] = env.objective.job_weights

    elif isinstance(env.objective, objectives.TotalCompletionTime):
        weighted = False
        weights = []

    else:
        raise ValueError(
            "Smith's rules symmetry breaking is only applicable"
            "for WeightedCompletionTime or TotalCompletionTime objectives."
        )

    machine_constraint: list[list[Task]] = [[] for _ in range(state.n_machines)]

    for task in state.tasks:
        for machine_id in task.machines:
            machine_constraint[machine_id].append(task)

    for machine_id, machine_tasks in enumerate(machine_constraint):
        priorities: list[tuple[float, Task]] = []
        for task in machine_tasks:
            if weighted:
                priority = -weights[task.job_id] / int(task.remaining_times[machine_id])
            else:
                priority = int(task.remaining_times[machine_id])
            
            priorities.append((priority, task))

        priorities.sort(key=lambda x: x[0])

        for idx, (_, task) in enumerate(priorities):
            S_j = lpSum(
                prev_task.remaining_times[machine_id] *
                decision_vars.assignments[prev_task.task_id][machine_id]
                for (_, prev_task) in priorities[:idx]
            )

            implication_pulp(
                model,
                antecedent=decision_vars.assignments[task.task_id][machine_id],
                consequent=(decision_vars.start_times[task.task_id], "==", S_j),
                big_m=int(
                    task.get_start_ub() - task.get_start_lb()
                ),
                name=f"SB_smiths_{task.task_id}_machine_{machine_id}",
            )

            for _, prev_task in priorities[:idx]:
                implication_pulp(
                    model,
                    antecedent=(
                        decision_vars.assignments[task.task_id][machine_id],
                        decision_vars.assignments[prev_task.task_id][machine_id],
                    ),
                    consequent=(
                        decision_vars.get_order(prev_task.task_id, task.task_id),
                        ">=",
                        1,
                    ),
                    big_m=1,
                    name=f"SB_smiths_order_{prev_task.task_id}_{task.task_id}_machine_{machine_id}",
                )


def tardiness_dominance_symmetry_breaking(
    env: SchedulingEnv, model: LpProblem, decision_vars: PulpSchedulingVariables
) -> None:
    "In agreeable jobs, break symmetry by ordering by tardiness dominance"
    state = env.state

    machine_constraint: list[list[Task]] = [[] for _ in range(state.n_machines)]

    for task in state.tasks:
        for machine_id in task.machines:
            machine_constraint[machine_id].append(task)

    if isinstance(env.objective, objectives.WeightedTardiness):
        weighted = True
        weights: list[float] = env.objective.job_weights
        due_dates = env.objective.due_dates

    elif isinstance(env.objective, objectives.TotalTardiness):
        weighted = False
        weights = []
        due_dates = env.objective.due_dates

    else:
        raise ValueError(
            "Tardiness dominance symmetry breaking is only applicable"
            "for WeightedTardiness or TotalTardiness objectives."
        )

    for machine_id, machine_tasks in enumerate(machine_constraint):
        sorted_tasks = sorted(
            machine_tasks, key=lambda task: due_dates[task.job_id]
        )

        # Necessarely d_i <= d_j
        for task_i, task_j in combinations(sorted_tasks, 2):
            if (
                task_i.remaining_times[machine_id] >= task_j.remaining_times[machine_id] or 
                (weighted and weights[task_i.job_id] <= weights[task_j.job_id])
            ):
                continue

            implication_pulp(
                model,
                antecedent=(
                    decision_vars.assignments[task_i.task_id][machine_id],
                    decision_vars.assignments[task_j.task_id][machine_id],
                ),
                consequent=(
                    decision_vars.get_order(task_i.task_id, task_j.task_id),
                    ">=",
                    1,
                ),
                big_m=1,
                name=f"SB_tardiness_order_{task_i.task_id}_{task_j.task_id}_machine_{machine_id}",
            )

            implication_pulp(
                model,
                antecedent=(
                    decision_vars.assignments[task_i.task_id][machine_id],
                    decision_vars.assignments[task_j.task_id][machine_id],
                ),
                consequent=(
                    decision_vars.end_times[task_i.task_id],
                    "<=",
                    decision_vars.start_times[task_j.task_id],
                ),
                big_m=int(task_i.get_end_ub() - task_j.get_start_lb()),
                name=f"SB_tardiness_{task_i.task_id}_{task_j.task_id}_machine_{machine_id}",
            )

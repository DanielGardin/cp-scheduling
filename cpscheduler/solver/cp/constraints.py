from itertools import combinations

from cpscheduler.environment import (
    BatchConstraint,
    ConstantProcessingTime,
    DeadlineConstraint,
    HorizonConstraint,
    MachineBreakdownConstraint,
    MachineConstraint,
    MachineEligibilityConstraint,
    NoWaitConstraint,
    NonOverlapConstraint,
    NonRenewableResourceConstraint,
    ORPrecedenceConstraint,
    OptionalityConstraint,
    PrecedenceConstraint,
    PreemptionConstraint,
    ReleaseDateConstraint,
    ResourceConstraint,
    SetupConstraint,
)
from cpscheduler.environment.state import ScheduleState

from .cp_formulation import DisjunctiveCPFormulation


DisjunctiveCPFormulation.mark_constraint_as_handled(
    ReleaseDateConstraint,
    DeadlineConstraint,
    HorizonConstraint,
    OptionalityConstraint,
    MachineEligibilityConstraint,
    ConstantProcessingTime,
    PreemptionConstraint,
)


def _expr(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def _horizon(state: ScheduleState) -> int:
    return max(state.get_end_ub(task_id) for task_id in range(state.n_tasks))


def _active_expr(start_expr: str, end_expr: str, time: int) -> str:
    return f"bool2int(({start_expr} <= {time}) /\\ ({time} < {end_expr}))"


@DisjunctiveCPFormulation.register_constraint(PrecedenceConstraint)
def precedence_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: PrecedenceConstraint,
) -> None:
    for task_id, children in constraint.children.items():
        for child_id in children:
            formulation.implication(
                antecedent=(formulation.presents[task_id], formulation.presents[child_id]),
                consequent=(
                    f"{formulation._end_expr(task_id)} <= {formulation._start_expr(child_id)}",
                )[0],
                name=f"precedence_{task_id}_{child_id}",
            )


@DisjunctiveCPFormulation.register_constraint(NoWaitConstraint)
def no_wait_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: NoWaitConstraint,
) -> None:
    for task_id, children in constraint.children.items():
        for child_id in children:
            formulation.implication(
                antecedent=(formulation.presents[task_id], formulation.presents[child_id]),
                consequent=(
                    f"{formulation._end_expr(task_id)} = {formulation._start_expr(child_id)}",
                )[0],
                name=f"no_wait_{task_id}_{child_id}",
            )


@DisjunctiveCPFormulation.register_constraint(ORPrecedenceConstraint)
def or_precedence_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: ORPrecedenceConstraint,
) -> None:
    for child_id, parents in constraint.parents.items():
        parent_ends = ", ".join(formulation._end_expr(parent_id) for parent_id in parents)
        formulation.add_constraint(
            f"{formulation._start_expr(child_id)} >= min([{parent_ends}])",
            f"or_precedence_{child_id}",
        )


@DisjunctiveCPFormulation.register_constraint(NonOverlapConstraint)
def non_overlap_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: NonOverlapConstraint,
) -> None:
    for group_id, group in enumerate(constraint.groups_map):
        tasks = sorted(group)
        for i, j in combinations(tasks, 2):
            order_var = formulation.get_order(i, j)
            presence_i = formulation._present_expr(i)
            presence_j = formulation._present_expr(j)

            formulation.implication(
                antecedent=(order_var, presence_i, presence_j),
                consequent=f"{formulation._end_expr(i)} <= {formulation._start_expr(j)}",
                name=f"non_overlap_{group_id}_{i}_{j}_ij",
            )
            formulation.implication(
                antecedent=(f"({order_var}) = false", presence_i, presence_j),
                consequent=f"{formulation._end_expr(j)} <= {formulation._start_expr(i)}",
                name=f"non_overlap_{group_id}_{i}_{j}_ji",
            )


@DisjunctiveCPFormulation.register_constraint(MachineConstraint)
def machine_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: MachineConstraint,
) -> None:
    for machine_id, tasks in enumerate(constraint.machine_map):
        task_ids = sorted(tasks)
        for i, j in combinations(task_ids, 2):
            order_var = formulation.get_order(i, j)
            assign_i = formulation._assignment_expr(i, machine_id)
            assign_j = formulation._assignment_expr(j, machine_id)

            formulation.implication(
                antecedent=(order_var, assign_i, assign_j),
                consequent=f"{formulation._end_expr(i)} <= {formulation._start_expr(j)}",
                name=f"machine_{machine_id}_{i}_{j}_ij",
            )
            formulation.implication(
                antecedent=(f"({order_var}) = false", assign_i, assign_j),
                consequent=f"{formulation._end_expr(j)} <= {formulation._start_expr(i)}",
                name=f"machine_{machine_id}_{i}_{j}_ji",
            )


@DisjunctiveCPFormulation.register_constraint(BatchConstraint)
def batch_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: BatchConstraint,
) -> None:
    horizon = _horizon(state)
    for machine_id, tasks in enumerate(constraint.machine_map):
        task_ids = sorted(tasks)
        capacity = int(constraint.capacity[machine_id])

        if capacity <= 1:
            for i, j in combinations(task_ids, 2):
                order_var = formulation.get_order(i, j)
                assign_i = formulation._assignment_expr(i, machine_id)
                assign_j = formulation._assignment_expr(j, machine_id)

                formulation.implication(
                    antecedent=(order_var, assign_i, assign_j),
                    consequent=f"{formulation._end_expr(i)} <= {formulation._start_expr(j)}",
                    name=f"batch_{machine_id}_{i}_{j}_ij",
                )
                formulation.implication(
                    antecedent=(f"({order_var}) = false", assign_i, assign_j),
                    consequent=f"{formulation._end_expr(j)} <= {formulation._start_expr(i)}",
                    name=f"batch_{machine_id}_{i}_{j}_ji",
                )

            continue

        for time in range(horizon + 1):
            terms = [
                f"bool2int({formulation._assignment_expr(task_id, machine_id)}) * {_active_expr(formulation._start_expr(task_id), formulation._end_expr(task_id), time)}"
                for task_id in task_ids
            ]

            if not terms:
                continue

            formulation.add_constraint(
                f"{' + '.join(terms)} <= {capacity}",
                f"batch_{machine_id}_{time}",
            )


@DisjunctiveCPFormulation.register_constraint(ResourceConstraint)
def resource_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: ResourceConstraint,
) -> None:
    task_ids = list(range(state.n_tasks))
    horizon = _horizon(state)

    for resource_id, capacity in enumerate(constraint.capacities):
        demand_terms = [
            (
                task_id,
                constraint.resources[resource_id][task_id],
            )
            for task_id in task_ids
            if constraint.resources[resource_id][task_id] > 0
        ]

        if not demand_terms:
            continue

        for time in range(horizon + 1):
            terms = [
                f"{demand} * bool2int({_expr(formulation.presents[task_id])}) * {_active_expr(formulation._start_expr(task_id), formulation._end_expr(task_id), time)}"
                for task_id, demand in demand_terms
            ]

            formulation.add_constraint(
                f"{' + '.join(terms)} <= {capacity}",
                f"resource_{resource_id}_{time}",
            )


@DisjunctiveCPFormulation.register_constraint(NonRenewableResourceConstraint)
def non_renewable_resource_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: NonRenewableResourceConstraint,
) -> None:
    for resource_id, capacity in enumerate(constraint.capacities):
        terms: list[str] = []
        for task_id, demand in enumerate(constraint.resources[resource_id]):
            if demand <= 0:
                continue

            terms.append(
                f"{demand} * bool2int({_expr(formulation.presents[task_id])})"
            )

        if not terms:
            continue

        formulation.add_constraint(
            f"{' + '.join(terms)} <= {capacity}",
            f"non_renewable_resource_{resource_id}",
        )


@DisjunctiveCPFormulation.register_constraint(SetupConstraint)
def setup_constraint(
    formulation: DisjunctiveCPFormulation,
    state: ScheduleState,
    constraint: SetupConstraint,
) -> None:
    processed: set[tuple[int, int]] = set()

    for task_id, setup_times in constraint.current_setup_times.items():
        for other_task_id, setup_time in setup_times.items():
            if task_id == other_task_id:
                continue

            pair = (task_id, other_task_id)
            if pair in processed or (other_task_id, task_id) in processed:
                continue

            processed.add(pair)
            setup_ij = int(setup_times.get(other_task_id, 0))
            setup_ji = int(
                constraint.current_setup_times.get(other_task_id, {}).get(task_id, 0)
            )

            if setup_ij > 0:
                formulation.implication(
                    antecedent=(
                        formulation.presents[task_id],
                        formulation.presents[other_task_id],
                        formulation.get_order(task_id, other_task_id),
                    ),
                    consequent=(
                        f"{formulation._start_expr(other_task_id)} >= {formulation._end_expr(task_id)} + {setup_ij}",
                    )[0],
                    name=f"setup_{task_id}_{other_task_id}",
                )

            if setup_ji > 0:
                formulation.implication(
                    antecedent=(
                        formulation.presents[task_id],
                        formulation.presents[other_task_id],
                        formulation.get_order(other_task_id, task_id),
                    ),
                    consequent=(
                        f"{formulation._start_expr(task_id)} >= {formulation._end_expr(other_task_id)} + {setup_ji}",
                    )[0],
                    name=f"setup_{other_task_id}_{task_id}",
                )

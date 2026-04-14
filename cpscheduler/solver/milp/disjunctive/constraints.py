from itertools import combinations

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.constraints import (
    PrecedenceConstraint,
    NoWaitConstraint,
    ResourceConstraint,
    NonRenewableResourceConstraint,
    MachineConstraint,
    SetupConstraint,
    # MachineBreakdownConstraint,
    NonOverlapConstraint,
    ReleaseDateConstraint,
    DeadlineConstraint,
    HorizonConstraint,
)

from cpscheduler.solver.milp.disjunctive.formulation import (
    DisjunctiveMILPFormulation,
)

DisjunctiveMILPFormulation.mark_constraint_as_handled(
    ReleaseDateConstraint,
    DeadlineConstraint,
    HorizonConstraint,
)


@DisjunctiveMILPFormulation.register_constraint(PrecedenceConstraint)
def prec_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: PrecedenceConstraint,
) -> None:
    for task_id, children in constraint.children.items():
        for child_id in children:
            formulation.add_constraint(
                formulation.end_times[task_id]
                <= formulation.start_times[child_id],
                f"precedence_{task_id}_{child_id}",
            )

            formulation.set_global_order(task_id, child_id)


@DisjunctiveMILPFormulation.register_constraint(NoWaitConstraint)
def no_wait_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: NoWaitConstraint,
) -> None:
    for task_id, children in constraint.children.items():
        for child_id in children:
            formulation.add_constraint(
                formulation.end_times[task_id]
                == formulation.start_times[child_id],
                f"no_wait_{task_id}_{child_id}",
            )

            formulation.set_global_order(task_id, child_id)


@DisjunctiveMILPFormulation.register_constraint(NonOverlapConstraint)
def non_overlap_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: NonOverlapConstraint,
) -> None:
    for group in constraint.groups_map:
        for i, j in combinations(group, 2):
            order_var = formulation.get_order(i, j)
            presence_i = formulation.present[i]
            presence_j = formulation.present[j]

            formulation.implication(
                antecedent=(
                    order_var,
                    presence_i,
                    presence_j,
                ),
                consequent=(
                    formulation.end_times[i],
                    "<=",
                    formulation.start_times[j],
                ),
                name=f"disjunctive_{i}_{j}_order",
            )

            formulation.implication(
                antecedent=(
                    1 - order_var,
                    presence_i,
                    presence_j,
                ),
                consequent=(
                    formulation.end_times[j],
                    "<=",
                    formulation.start_times[i],
                ),
                name=f"disjunctive_{j}_{i}_order",
            )


@DisjunctiveMILPFormulation.register_constraint(MachineConstraint)
def machine_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: MachineConstraint,
) -> None:
    for machine_id, tasks in enumerate(constraint.machine_map):
        for i, j in combinations(tasks, 2):
            order_var = formulation.get_order(i, j)
            assignments_i = formulation.assignments[i]
            assignments_j = formulation.assignments[j]

            formulation.implication(
                antecedent=(
                    order_var,
                    assignments_i[machine_id],
                    assignments_j[machine_id],
                ),
                consequent=(
                    formulation.end_times[i],
                    "<=",
                    formulation.start_times[j],
                ),
                name=f"machine_{machine_id}_order_{i}_{j}",
            )

            formulation.implication(
                antecedent=(
                    1 - order_var,
                    assignments_i[machine_id],
                    assignments_j[machine_id],
                ),
                consequent=(
                    formulation.end_times[j],
                    "<=",
                    formulation.start_times[i],
                ),
                name=f"machine_{machine_id}_order_{j}_{i}",
            )


@DisjunctiveMILPFormulation.register_constraint(ResourceConstraint)
def resource_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: ResourceConstraint,
) -> None:
    # Resource constraints are actually quite tricky to model in a disjunctive
    # formulation.
    raise NotImplementedError(
        "Resource constraints are not available in the disjunctive formulation at the moment."
    )


@DisjunctiveMILPFormulation.register_constraint(NonRenewableResourceConstraint)
def non_renewable_resource_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: NonRenewableResourceConstraint,
) -> None:
    for resource_id, capacity in enumerate(constraint.capacities):
        resource_demand = sum(
            demand * formulation.present[task_id]
            for task_id, demand in enumerate(constraint.resources[resource_id])
            if demand > 0
        )

        formulation.add_constraint(
            resource_demand <= capacity,
            f"non_renewable_resource_{resource_id}",
        )


@DisjunctiveMILPFormulation.register_constraint(SetupConstraint)
def setup_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: SetupConstraint,
) -> None:
    for task_id, setup_times in constraint.current_setup_times.items():
        for other_task_id, setup_time in setup_times.items():
            time = int(setup_time)

            if time > 0:
                order_var = formulation.get_order(task_id, other_task_id)

                formulation.implication(
                    antecedent=(
                        order_var,
                        formulation.present[task_id],
                        formulation.present[other_task_id],
                    ),
                    consequent=(
                        formulation.start_times[other_task_id],
                        ">=",
                        formulation.end_times[task_id] + time,
                    ),
                    name=f"setup_{task_id}_{other_task_id}",
                )

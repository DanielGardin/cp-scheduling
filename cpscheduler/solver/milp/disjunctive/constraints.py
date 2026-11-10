"""Module for defining constraints for the disjunctive MILP formulation."""

from itertools import combinations

from cpscheduler.environment.constraints import (
    AllOrNothingConstraint,
    AtMostOneConstraint,
    DeadlineConstraint,
    ExactlyOneConstraint,
    HorizonConstraint,
    MachineConstraint,
    # MachineBreakdownConstraint,
    NonOverlapConstraint,
    NonRenewableResourceConstraint,
    NoWaitConstraint,
    PrecedenceConstraint,
    RejectableConstraint,
    ReleaseDateConstraint,
    ResourceConstraint,
    SetupConstraint,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.solver.milp.disjunctive.formulation import (
    DisjunctiveMILPFormulation,
)
from cpscheduler.solver.milp.pyomo_formulation import PYOMO_PARAM

DisjunctiveMILPFormulation.mark_constraint_as_handled(
    ReleaseDateConstraint,
    DeadlineConstraint,
    HorizonConstraint,
)


@DisjunctiveMILPFormulation.register_constraint(PrecedenceConstraint)
def _prec_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: PrecedenceConstraint,
) -> None:
    precedence_name = constraint.parents.name

    for task_id, children in constraint.children.items():
        for child_id in children:
            formulation.add_constraint(
                formulation.end_times[task_id]
                <= formulation.start_times[child_id],
                f"{precedence_name}_{task_id}_{child_id}",
            )

            formulation.set_global_order(task_id, child_id)


@DisjunctiveMILPFormulation.register_constraint(NoWaitConstraint)
def _no_wait_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: NoWaitConstraint,
) -> None:
    precedence_name = constraint.parents.name

    for task_id, children in constraint.children.items():
        for child_id in children:
            formulation.add_constraint(
                formulation.end_times[task_id]
                == formulation.start_times[child_id],
                f"{precedence_name}_{task_id}_{child_id}",
            )

            formulation.set_global_order(task_id, child_id)


def _disjunctive_pair(
    formulation: DisjunctiveMILPFormulation,
    i: int,
    j: int,
    shared: tuple[PYOMO_PARAM, ...],
    name: str,
) -> None:
    """Separate i and j in time whenever both literals of `shared` hold.

    The orders are local to the shared resource: `order(i, j)` and
    `order(j, i)` are independent variables, and the clause only asks for one
    of them when i and j actually compete for the resource.
    """
    # Tasks that can never share the resource are never ordered
    if any(formulation.is_true(literal == 1) is False for literal in shared):
        return

    i_first = formulation.get_order(i, j)
    j_first = formulation.get_order(j, i)

    formulation.implication(
        antecedent=(i_first, *shared),
        consequent=(
            formulation.end_times[i],
            "<=",
            formulation.start_times[j],
        ),
        name=f"{name}_{i}_{j}",
    )

    formulation.implication(
        antecedent=(j_first, *shared),
        consequent=(
            formulation.end_times[j],
            "<=",
            formulation.start_times[i],
        ),
        name=f"{name}_{j}_{i}",
    )

    formulation.clause(
        (i_first, j_first, *(1 - literal for literal in shared)),
        name=f"{name}_{i}_{j}_cover",
    )


@DisjunctiveMILPFormulation.register_constraint(NonOverlapConstraint)
def _non_overlap_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: NonOverlapConstraint,
) -> None:
    for group_id, group in enumerate(constraint.groups.value):
        for i, j in combinations(group, 2):
            _disjunctive_pair(
                formulation,
                i,
                j,
                (formulation.present[i], formulation.present[j]),
                f"disjunctive_{group_id}",
            )


@DisjunctiveMILPFormulation.register_constraint(MachineConstraint)
def _machine_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: MachineConstraint,
) -> None:
    for machine_id, tasks in enumerate(constraint.machine_map):
        for i, j in combinations(tasks, 2):
            _disjunctive_pair(
                formulation,
                i,
                j,
                (
                    formulation.present[i],
                    formulation.assignments[i][machine_id],
                    formulation.present[j],
                    formulation.assignments[j][machine_id],
                ),
                f"machine_{machine_id}",
            )


@DisjunctiveMILPFormulation.register_constraint(ResourceConstraint)
def _resource_constraint(
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
def _non_renewable_resource_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: NonRenewableResourceConstraint,
) -> None:
    resource_demand = sum(
        demand * formulation.present[task_id]
        for task_id, demand in enumerate(constraint.resources.value)
        if demand > 0
    )

    formulation.add_constraint(
        resource_demand <= constraint.capacity.value,
        f"non_renewable_resource_{constraint.resources.name}",
    )


@DisjunctiveMILPFormulation.register_constraint(SetupConstraint)
def _setup_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: SetupConstraint,
) -> None:
    # Setup times only separate tasks processed in the same machine
    for task_id, setup_times in constraint.current_setup_times.items():
        for other_task_id, setup_time in setup_times.items():
            time = int(setup_time)

            if time <= 0:
                continue

            shared_machines = set(state.get_machines(task_id)) & set(
                state.get_machines(other_task_id)
            )

            if not shared_machines:
                continue

            order_var = formulation.get_order(task_id, other_task_id)

            for machine_id in shared_machines:
                formulation.implication(
                    antecedent=(
                        order_var,
                        formulation.present[task_id],
                        formulation.present[other_task_id],
                        formulation.assignments[task_id][machine_id],
                        formulation.assignments[other_task_id][machine_id],
                    ),
                    consequent=(
                        formulation.start_times[other_task_id],
                        ">=",
                        formulation.end_times[task_id] + time,
                    ),
                    name=f"setup_{task_id}_{other_task_id}_{machine_id}",
                )


def _group_presence(
    formulation: DisjunctiveMILPFormulation, tasks: list[int]
) -> PYOMO_PARAM | None:
    """Return the number of present tasks, or None when all are decided.

    A decided group was already checked by the constraint itself on reset.
    """
    literals = [formulation.present[task_id] for task_id in tasks]

    if all(
        formulation.is_true(literal == 1) is not None for literal in literals
    ):
        return None

    return sum(literals)


@DisjunctiveMILPFormulation.register_constraint(AtMostOneConstraint)
def _at_most_one_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: AtMostOneConstraint,
) -> None:
    for group_id, tasks in enumerate(constraint.task_groups):
        n_present = _group_presence(formulation, tasks)

        if n_present is not None:
            formulation.add_constraint(
                n_present <= 1, f"at_most_one_{group_id}"
            )


@DisjunctiveMILPFormulation.register_constraint(ExactlyOneConstraint)
def _exactly_one_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: ExactlyOneConstraint,
) -> None:
    for group_id, tasks in enumerate(constraint.task_groups):
        n_present = _group_presence(formulation, tasks)

        if n_present is not None:
            formulation.add_constraint(
                n_present == 1, f"exactly_one_{group_id}"
            )


@DisjunctiveMILPFormulation.register_constraint(AllOrNothingConstraint)
def _all_or_nothing_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: AllOrNothingConstraint,
) -> None:
    for group_id, tasks in enumerate(constraint.task_groups):
        if not tasks:
            continue

        first = formulation.present[tasks[0]]
        for task_id in tasks[1:]:
            present = formulation.present[task_id]
            formulation.add_constraint(
                present == first, f"all_or_nothing_{group_id}_{task_id}"
            )


@DisjunctiveMILPFormulation.register_constraint(RejectableConstraint)
def _rejectable_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    constraint: RejectableConstraint,
) -> None:
    _all_or_nothing_constraint(formulation, state, constraint)

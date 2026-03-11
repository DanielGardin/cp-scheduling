from pulp import LpProblem, lpSum

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.schedule_setup import (
    SingleMachineSetup,
    IdenticalParallelMachineSetup,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
    JobShopSetup,
    FlowShopSetup,
    OpenShopSetup,
)

from cpscheduler.solver.milp.pulp_utils import PULP_PARAM, pulp_add_constraint

from cpscheduler.solver.milp.disjunctive.formulation import (
    DisjunctiveMILPFormulation,
)


def non_preemptive_constraint(
    model: LpProblem,
    start_time: PULP_PARAM,
    end_time: PULP_PARAM,
    processing_time: PULP_PARAM,
    task_id: int,
) -> None:
    pulp_add_constraint(
        model,
        end_time == start_time + processing_time,
        f"non_preemptive_{task_id}",
    )


def assignment_constraint(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
) -> None:
    for task_id in range(state.n_tasks):
        pulp_add_constraint(
            formulation.model,
            lpSum(formulation.assignments[task_id]) == 1,
            f"assignment_{task_id}",
        )


@DisjunctiveMILPFormulation.register_setup(SingleMachineSetup)
def single_machine_setup(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    setup: SingleMachineSetup,
) -> None:
    for task_id in range(state.n_tasks):
        processing_time = state.get_remaining_time(task_id, 0)

        non_preemptive_constraint(
            formulation.model,
            formulation.start_times[task_id],
            formulation.end_times[task_id],
            processing_time,
            task_id,
        )


@DisjunctiveMILPFormulation.register_setup(IdenticalParallelMachineSetup)
def identical_parallel_machine_setup(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    setup: IdenticalParallelMachineSetup,
) -> None:
    assignment_constraint(formulation, state)

    for task_id in range(state.n_tasks):
        processing_time = state.get_remaining_time(task_id, machine_id=0)

        non_preemptive_constraint(
            formulation.model,
            formulation.start_times[task_id],
            formulation.end_times[task_id],
            processing_time,
            task_id,
        )


@DisjunctiveMILPFormulation.register_setup(UniformParallelMachineSetup)
def parallel_machine_setup(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    setup: UniformParallelMachineSetup | UnrelatedParallelMachineSetup,
) -> None:
    assignment_constraint(formulation, state)

    for task_id in range(state.n_tasks):
        machine_assignments = formulation.assignments[task_id]

        processing_time = lpSum(
            assignment * int(state.get_remaining_time(task_id, machine_id))
            for machine_id, assignment in enumerate(machine_assignments)
        )

        non_preemptive_constraint(
            formulation.model,
            formulation.start_times[task_id],
            formulation.end_times[task_id],
            processing_time,
            task_id,
        )


@DisjunctiveMILPFormulation.register_setup(UnrelatedParallelMachineSetup)
def unrelated_parallel_machine_setup(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    setup: UnrelatedParallelMachineSetup,
) -> None:
    assignment_constraint(formulation, state)

    for task_id in range(state.n_tasks):
        machine_assignments = formulation.assignments[task_id]

        processing_time = lpSum(
            assignment * int(state.get_remaining_time(task_id, machine_id))
            for machine_id, assignment in enumerate(machine_assignments)
        )

        non_preemptive_constraint(
            formulation.model,
            formulation.start_times[task_id],
            formulation.end_times[task_id],
            processing_time,
            task_id,
        )


@DisjunctiveMILPFormulation.register_setup(FlowShopSetup)
@DisjunctiveMILPFormulation.register_setup(JobShopSetup)
def job_shop_setup(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    setup: JobShopSetup,
) -> None:
    for task_id in range(state.n_tasks):
        machine_id = state.get_machines(task_id)[0]
        processing_time = state.get_remaining_time(task_id, machine_id)

        non_preemptive_constraint(
            formulation.model,
            formulation.start_times[task_id],
            formulation.end_times[task_id],
            processing_time,
            task_id,
        )


@DisjunctiveMILPFormulation.register_setup(OpenShopSetup)
def open_shop_setup(
    formulation: DisjunctiveMILPFormulation,
    state: ScheduleState,
    setup: OpenShopSetup,
) -> None:
    for task_id in range(state.n_tasks):
        machine_id = state.get_machines(task_id)[0]
        processing_time = state.get_remaining_time(task_id, machine_id)

        non_preemptive_constraint(
            formulation.model,
            formulation.start_times[task_id],
            formulation.end_times[task_id],
            processing_time,
            task_id,
        )

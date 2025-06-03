from typing import Callable

from pulp import LpProblem, lpSum

from functools import singledispatch

from ...environment.schedule_setup import (
    ScheduleSetup,
    SingleMachineSetup,
    IdenticalParallelMachineSetup,
    UniformParallelMachineSetup,
    JobShopSetup,
    OpenShopSetup
)

from .tasks import PulpVariables

@singledispatch
def export_setup_pulp(
    setup: ScheduleSetup,
) -> Callable[[LpProblem, PulpVariables], None]:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


@export_setup_pulp.register
def _(setup: SingleMachineSetup) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(
                decision_vars.end_times[task_id] ==
                decision_vars.start_times[task_id] + task.processing_times[0],
                f"non_preemptive_{task_id}"
            )

    return export_model

@export_setup_pulp.register
def _(setup: IdenticalParallelMachineSetup) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(
                decision_vars.end_times[task_id] ==
                decision_vars.start_times[task_id] + task.processing_times[0],
                f"non_preemptive_{task_id}"
            )

            model.addConstraint(lpSum(
                decision_vars.assignments[task_id][machine_id]
                for machine_id in range(setup.n_machines)
                ) == 1,
                f"assignment_{task_id}"
            )

    return export_model

# TODO: Linearize the assignment variables for uniform parallel machines
@export_setup_pulp.register
def _(setup: UniformParallelMachineSetup) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
        return

    return export_model

@export_setup_pulp.register
def _(setup: JobShopSetup) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(
                decision_vars.end_times[task_id] ==
                decision_vars.start_times[task_id] + task.processing_times[setup.get_machine(task_id)]
            )

    return export_model

@export_setup_pulp.register
def _(setup: OpenShopSetup) -> Callable[[LpProblem, PulpVariables], None]:
    def export_model(model: LpProblem, decision_vars: PulpVariables) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(
                decision_vars.end_times[task_id] ==
                decision_vars.start_times[task_id] + task.processing_times[setup.get_machine(task_id)]
            )

    return export_model
from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, lpSum

from functools import singledispatch

from ...environment.schedule_setup import (
    ScheduleSetup,
    SingleMachineSetup,
    IdenticalParallelMachineSetup,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
    JobShopSetup,
    OpenShopSetup
)

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable

ModelExport: TypeAlias = Callable[[LpProblem], None]

@singledispatch
def export_setup_pulp(setup: ScheduleSetup, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


@export_setup_pulp.register
def _(setup: SingleMachineSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(
                variables.end_times[task_id] ==
                variables.start_times[task_id] + task.processing_times[0],
                f"non_preemptive_{task_id}"
            )

    return export_model

@export_setup_pulp.register
def _(setup: IdenticalParallelMachineSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(
                variables.end_times[task_id] ==
                variables.start_times[task_id] + task.processing_times[0],
                f"non_preemptive_{task_id}"
            )

            model.addConstraint(lpSum(
                variables.assignments[task_id][machine_id]
                for machine_id in range(setup.n_machines)
                ) == 1,
                f"assignment_{task_id}"
            )

    return export_model

@export_setup_pulp.register
def _(setup: UniformParallelMachineSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(lpSum(
                variables.assignments[task_id][machine_id]
                for machine_id in range(setup.n_machines)
                ) == 1,
                f"assignment_{task_id}"
            )

            processing_time = lpSum(
                variables.assignments[task_id][machine_id] * task.processing_times[machine_id]
                for machine_id in task.processing_times
            )


    return export_model

@export_setup_pulp.register
def _(setup: UnrelatedParallelMachineSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        return

    return export_model

@export_setup_pulp.register
def _(setup: JobShopSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(
                variables.end_times[task_id] ==
                variables.start_times[task_id] + task.processing_times[setup.get_machine(task_id)]
            )

    return export_model

@export_setup_pulp.register
def _(setup: OpenShopSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem) -> None:
        for task_id, task in enumerate(setup.tasks):
            model.addConstraint(
                variables.end_times[task_id] ==
                variables.start_times[task_id] + task.processing_times[setup.get_machine(task_id)]
            )

    return export_model
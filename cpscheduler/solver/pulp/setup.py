from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, lpSum

from multimethod import multidispatch

from cpscheduler.environment.tasks import Tasks
from cpscheduler.environment.data import SchedulingData
from cpscheduler.environment.schedule_setup import (
    ScheduleSetup,
    SingleMachineSetup,
    IdenticalParallelMachineSetup,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
    JobShopSetup,
    OpenShopSetup,
)

from .pulp_utils import PULP_PARAM, PULP_EXPRESSION, pulp_add_constraint
from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable

ModelExport: TypeAlias = Callable[[LpProblem, Tasks, SchedulingData], None]


def non_preemptive_constraint(
    model: LpProblem,
    start_time: PULP_EXPRESSION | int,
    end_time: PULP_EXPRESSION | int,
    processing_time: PULP_PARAM,
    task_id: int,
) -> None:
    pulp_add_constraint(
        model, end_time == start_time + processing_time, f"non_preemptive_{task_id}"
    )


@multidispatch
def export_setup_pulp(setup: ScheduleSetup, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


# Timetables do not require any specific setup constraints:
# Their structure already compiles the necessary task information
# gathered at the setup stage.
@export_setup_pulp.register
def _(setup: ScheduleSetup, variables: PulpTimetable) -> ModelExport:
    return lambda model, tasks, data: None


@export_setup_pulp.register
def _(setup: SingleMachineSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for task_id, task in enumerate(tasks):
            processing_time = task.processing_times[0]

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

    return export_model


@export_setup_pulp.register
def _(
    setup: IdenticalParallelMachineSetup, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for task_id, task in enumerate(tasks):
            pulp_add_constraint(
                model,
                lpSum(machine for machine in variables.assignments[task_id]) == 1,
                f"assignment_{task_id}",
            )

            processing_time = task.processing_times[0]

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

    return export_model


@export_setup_pulp.register
def _(
    setup: UniformParallelMachineSetup, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for task_id, task in enumerate(tasks):
            pulp_add_constraint(
                model,
                lpSum(machine for machine in variables.assignments[task_id]) == 1,
                f"assignment_{task_id}",
            )

            processing_time = lpSum(
                variables.assignments[task_id][machine_id] * processing_time
                for machine_id, processing_time in task.processing_times.items()
            )

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

    return export_model


@export_setup_pulp.register
def _(
    setup: UnrelatedParallelMachineSetup, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for task_id, task in enumerate(tasks):
            pulp_add_constraint(
                model,
                lpSum(machine for machine in variables.assignments[task_id]) == 1,
                f"assignment_{task_id}",
            )

            processing_time = lpSum(
                variables.assignments[task_id][machine_id] * processing_time
                for machine_id, processing_time in task.processing_times.items()
            )

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

    return export_model


@export_setup_pulp.register
def _(setup: JobShopSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for task_id, task in enumerate(tasks):
            processing_time = next(iter(task.processing_times.values()))

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

    return export_model


@export_setup_pulp.register
def _(setup: OpenShopSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks, data: SchedulingData) -> None:
        for task_id, task in enumerate(tasks):
            processing_time = next(iter(task.processing_times.values()))

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

    return export_model

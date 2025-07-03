from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, lpSum, LpVariable

from multimethod import multidispatch

from cpscheduler.environment.tasks import Tasks
from cpscheduler.environment.schedule_setup import (
    ScheduleSetup,
    SingleMachineSetup,
    IdenticalParallelMachineSetup,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
    JobShopSetup,
    OpenShopSetup,
)

from .common import PULP_PARAM
from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable

ModelExport: TypeAlias = Callable[[LpProblem, Tasks], None]


def non_preemptive_constraint(
    model: LpProblem,
    start_time: LpVariable,
    end_time: LpVariable,
    processing_time: PULP_PARAM,
    task_id: int,
) -> None:
    model.addConstraint(
        end_time == start_time + processing_time, f"non_preemptive_{task_id}"
    )


@multidispatch
def export_setup_pulp(setup: ScheduleSetup, variables: PulpVariables) -> ModelExport:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


# Timetables do not require any specific setup constraints:
# Their structure already compiles the necessary task information
# gathered at the setup stage.
@export_setup_pulp.register
def _(setup: ScheduleSetup, variables: PulpTimetable) -> ModelExport:
    return lambda model, tasks: None


@export_setup_pulp.register
def _(setup: SingleMachineSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        for task_id, task in enumerate(tasks):
            processing_time = task.processing_times[0]

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

            model.addConstraint(
                variables.assignments[task_id][0] == 1, f"assignment_{task_id}"
            )

    return export_model


@export_setup_pulp.register
def _(
    setup: IdenticalParallelMachineSetup, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        for task_id, task in enumerate(tasks):
            processing_time = task.processing_times[0]

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

            model.addConstraint(
                lpSum(
                    variables.assignments[task_id][machine_id]
                    for machine_id in range(setup.n_machines)
                )
                == 1,
                f"assignment_{task_id}",
            )

    return export_model


@export_setup_pulp.register
def _(
    setup: UniformParallelMachineSetup, variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        for task_id, task in enumerate(tasks):
            model.addConstraint(
                lpSum(
                    variables.assignments[task_id][machine_id]
                    for machine_id in task.processing_times
                )
                == 1,
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
    def export_model(model: LpProblem, tasks: Tasks) -> None:
        for task_id, task in enumerate(tasks):
            model.addConstraint(
                lpSum(
                    variables.assignments[task_id][machine_id]
                    for machine_id in task.processing_times
                )
                == 1,
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
    def export_model(model: LpProblem, tasks: Tasks) -> None:
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
    def export_model(model: LpProblem, tasks: Tasks) -> None:
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

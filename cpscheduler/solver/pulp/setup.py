from typing import TypeAlias
from collections.abc import Callable

from pulp import LpProblem, lpSum

from multimethod import multidispatch

from cpscheduler.environment.state import ScheduleState
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

ModelExport: TypeAlias = Callable[[LpProblem, ScheduleState], None]

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
    raise NotImplementedError(
        f"Setup {setup} for variable {variables} not implemented for PuLP."
    )


# Timetables do not require any specific setup constraints:
# Their structure already compiles the necessary task information
# gathered at the setup stage.
@export_setup_pulp.register
def _(setup: ScheduleSetup, variables: PulpTimetable) -> ModelExport:
    return lambda model, state: None


# @export_setup_pulp.register
# def _(setup: SingleMachineSetup, variables: PulpSchedulingVariables) -> ModelExport:
#     def export_model(model: LpProblem, state: ScheduleState) -> None:
#         for task_id, task in enumerate(state.tasks):
#             processing_time = task.remaining_times[0]

#             non_preemptive_constraint(
#                 model,
#                 variables.start_times[task_id],
#                 variables.end_times[task_id],
#                 processing_time,
#                 task_id,
#             )

#     return export_model


# @export_setup_pulp.register
# def _(
#     setup: IdenticalParallelMachineSetup, variables: PulpSchedulingVariables
# ) -> ModelExport:
#     def export_model(model: LpProblem, state: ScheduleState) -> None:
#         for task_id, task in enumerate(state.tasks):
#             pulp_add_constraint(
#                 model,
#                 lpSum(machine for machine in variables.assignments[task_id]) == 1,
#                 f"assignment_{task_id}",
#             )

#             processing_time = task.remaining_times[0]

#             non_preemptive_constraint(
#                 model,
#                 variables.start_times[task_id],
#                 variables.end_times[task_id],
#                 processing_time,
#                 task_id,
#             )

#     return export_model


@export_setup_pulp.register
def _(
    setup: UniformParallelMachineSetup | UnrelatedParallelMachineSetup | IdenticalParallelMachineSetup | SingleMachineSetup,
    variables: PulpSchedulingVariables
) -> ModelExport:
    def export_model(model: LpProblem, state: ScheduleState) -> None:
        for task_id, task in enumerate(state.tasks):
            pulp_add_constraint(
                model,
                lpSum(machine for machine in variables.assignments[task_id]) == 1,
                f"assignment_{task_id}",
            )

            processing_time = lpSum(
                variables.assignments[task_id][machine_id] * processing_time
                for machine_id, processing_time in task.remaining_times.items()
            )

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

    return export_model


# @export_setup_pulp.register
# def _(
#     setup: UnrelatedParallelMachineSetup, variables: PulpSchedulingVariables
# ) -> ModelExport:
#     def export_model(model: LpProblem, state: ScheduleState) -> None:
#         for task_id, task in enumerate(state.tasks):
#             pulp_add_constraint(
#                 model,
#                 lpSum(machine for machine in variables.assignments[task_id]) == 1,
#                 f"assignment_{task_id}",
#             )

#             processing_time = lpSum(
#                 variables.assignments[task_id][machine_id] * processing_time
#                 for machine_id, processing_time in task.remaining_times.items()
#             )

#             non_preemptive_constraint(
#                 model,
#                 variables.start_times[task_id],
#                 variables.end_times[task_id],
#                 processing_time,
#                 task_id,
#             )

#     return export_model


@export_setup_pulp.register
def _(setup: JobShopSetup, variables: PulpSchedulingVariables) -> ModelExport:
    def export_model(model: LpProblem, state: ScheduleState) -> None:
        for task_id, task in enumerate(state.tasks):
            processing_time = next(iter(task.remaining_times.values()))

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
    def export_model(model: LpProblem, state: ScheduleState) -> None:
        for task_id, task in enumerate(state.tasks):
            processing_time = next(iter(task.remaining_times.values()))

            non_preemptive_constraint(
                model,
                variables.start_times[task_id],
                variables.end_times[task_id],
                processing_time,
                task_id,
            )

    return export_model

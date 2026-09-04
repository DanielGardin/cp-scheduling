"""Trajectory-tracing utilities for the environment."""

from copy import deepcopy

from typing_extensions import TypedDict, override

from cpscheduler.environment.backend import Instruction, ScheduleBackend
from cpscheduler.environment.constants import MachineID, TaskID
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.tracer.base import S, Tracer


class FullTrajectoryTracer(Tracer):
    """Tracer that records the trajectory of the environment.

    This tracer records the state and action at each decision step, allowing for
    a complete reconstruction of the environment's trajectory.
    Intended for debugging, offline analysis and trajectory replay.
    Note that the state is entirely copied into the tracer for each decision
    point, which may lead to a high memory usage and performance drops.
    """

    tracer_name = "full_trajectory"

    trajectory: list[tuple[Instruction, ScheduleState, ScheduleBackend]]

    def __init__(self) -> None:
        self.trajectory = []

    @override
    def reset(self, state: ScheduleState) -> None:
        self.trajectory.clear()

    @override
    def step(
        self,
        action: Instruction[S],
        state: ScheduleState,
        backend: S,
    ) -> None:
        self.trajectory.append(
            (deepcopy(action), deepcopy(state), deepcopy(backend))
        )


class _TrajectoryExport(TypedDict):
    tasks: list[TaskID]
    machines: list[MachineID]
    eligible_tasks: list[list[TaskID]]


class TrajectoryTracer(Tracer):
    """
    Tracer that records the trajectory of execution instructions.

    This tracer tracks the decision order of tasks in machines and recovering the
    eligible set.
    In order to an instruction to be logged, it must implement the `semantic`
    method and return the following information:
    - type: Must equal a string "execution"
    - task: The task id of the executing task
    - machine: The machine id of the executing task

    """

    tracer_name = "trajectory"

    tasks: list[TaskID]
    machines: list[MachineID]
    eligible: list[list[TaskID]]

    def __init__(self) -> None:
        self.tasks = []
        self.machines = []
        self.eligible = []

    @override
    def reset(self, state: ScheduleState) -> None:
        self.tasks.clear()
        self.machines.clear()
        self.eligible.clear()

    @override
    def step(
        self,
        action: Instruction[S],
        state: ScheduleState,
        backend: S,
    ) -> None:
        metadata = action.semantic(state, backend)

        if (
            metadata
            and metadata.get("type") == "execution"
            and "task" in metadata
            and "machine" in metadata
        ):
            task_id = TaskID(metadata["task"])
            machine_id = MachineID(metadata["machine"])

            self.tasks.append(task_id)
            self.machines.append(machine_id)
            self.eligible.append(backend.get_eligible_set(state))

    def export(self) -> _TrajectoryExport:
        """Export the tracer's internal state to a serializable format."""
        return {
            "tasks": self.tasks,
            "machines": self.machines,
            "eligible_tasks": self.eligible,
        }

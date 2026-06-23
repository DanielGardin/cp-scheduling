"""Trajectory-tracing utilities for the environment."""

from copy import deepcopy

from typing_extensions import override

from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.des import ExecuteEvent, SimulationEvent
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.tracer.base import Tracer


class FullTrajectoryTracer(Tracer):
    """Tracer that records the trajectory of the environment.

    This tracer records the state and action at each decision step, allowing for
    a complete reconstruction of the environment's trajectory.
    Intended for debugging, offline analysis and trajectory replay.
    Note that the state is entirely copied into the tracer for each decision
    point, which may lead to a high memory usage and performance drops.
    """

    tracer_name = "full_trajectory"

    trajectory: list[tuple[ScheduleState, SimulationEvent]]

    def __init__(self) -> None:
        self.trajectory = []

    @override
    def reset(self, state: ScheduleState) -> None:
        self.trajectory.clear()

    @override
    def step(self, state: ScheduleState, action: SimulationEvent) -> None:
        self.trajectory.append((deepcopy(state), deepcopy(action)))


class ExecutionTrajectoryTracer(Tracer):
    """Tracer that records a partial trajectory of the environment.

    This tracker only logs the decision outcomes when executing a task, including
    the task id, the machine id, the current time and the set of available tasks
    at the time of the decision.
    """

    tracer_name = "execution_trajectory"

    trajectory: list[tuple[TaskID, MachineID, Time, list[TaskID]]]

    def __init__(self) -> None:
        self.trajectory = []

    @override
    def reset(self, state: ScheduleState) -> None:
        self.trajectory.clear()

    @override
    def step(self, state: ScheduleState, action: SimulationEvent) -> None:
        if isinstance(action, ExecuteEvent):
            available_tasks = state.get_available_tasks()
            self.trajectory.append(
                (
                    action.task_id,
                    action.resolve_machine(state),
                    state.time,
                    available_tasks,
                )
            )

    @override
    def export(self) -> list[tuple[TaskID, MachineID, Time, list[TaskID]]]:
        return self.trajectory

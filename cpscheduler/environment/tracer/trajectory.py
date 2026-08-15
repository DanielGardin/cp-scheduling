"""Trajectory-tracing utilities for the environment."""

from copy import deepcopy

from typing_extensions import override

from cpscheduler.environment.backend import Instruction, ScheduleBackend
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

    trajectory: list[tuple[Instruction, ScheduleState, ScheduleBackend]]

    def __init__(self) -> None:
        self.trajectory = []

    @override
    def reset(self, state: ScheduleState) -> None:
        self.trajectory.clear()

    @override
    def step(
        self,
        action: Instruction,
        state: ScheduleState,
        backend: ScheduleBackend,
    ) -> None:
        self.trajectory.append(
            (deepcopy(action), deepcopy(state), deepcopy(backend))
        )

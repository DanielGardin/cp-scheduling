"""Base tracer class for the CPScheduler environment."""

from typing import Any, ClassVar, TypeVar

from cpscheduler.environment.backend import Instruction, ScheduleBackend
from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state import ScheduleState

S = TypeVar("S", bound=ScheduleBackend)


class Tracer(EzPickle):
    """Base class for tracers in the CPScheduler environment.

    Tracers are called before each decision step, dispatched by the backend.
    They are often used as a snapshot of the environment's internal state right
    before an action, used in learning planning policies, for example.
    """

    # Name of the tracer, used for identification and export.
    tracer_name: ClassVar[str] = "tracer"

    def initialize(self, instance: ProblemInstance) -> None:
        """Initialize the tracer with the environment's initial state."""

    def reset(self, state: ScheduleState) -> None:
        """Reset the tracer to its initial state."""

    def step(
        self,
        action: Instruction[S],
        state: ScheduleState,
        backend: S,
    ) -> None:
        """Process a step in the tracer.

        During each step, the tracer can access the current state and action in
        a read-only manner, attempting to change the state, or the action will
        lead to undefined behavior.
        """

    def export(self) -> dict[str, Any] | Any | None:
        """Export the tracer's internal state to a serializable format."""
        return None

"""Core components for the Discrete Event Simulation (DES) of the environment.

This module contains the main types and utilities for managing the event queue
and processing events according to their timing and blocking behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeAlias

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle, Time

if TYPE_CHECKING:
    from cpscheduler.environment.des.schedule import Schedule
    from cpscheduler.environment.state import ScheduleState

instructions: dict[str, type[SimulationEvent]] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class SimulationEvent(EzPickle):
    """Base class for all events in the simulation.

    Events are result from processing instructions or from the environment
    itself, triggering changes in the schedule state.

    To create a new event, subclass this class to define the instruction type
    and behavior in the simulation.

    Attributes
    ----------
    blocking: bool
        Whether this event blocks the processing of subsequent events. Blocking
        events are processed in a separate phase after all non-blocking events at
        the same time step, and they can delay the processing of subsequent events
        until they are resolved.

    """

    blocking: ClassVar[bool] = False

    def resolve(self, state: ScheduleState) -> SimulationEvent:
        """Resolve and statically validate the event."""
        return self

    # This are only used for C events
    def earliest_time(self, state: ScheduleState) -> Time:
        """Calculate the earliest time this event can be processed, given the current state."""
        return state.time

    def is_ready(self, state: ScheduleState) -> bool:
        """Check if the event is ready to be processed, given the current state."""
        return True

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        """Process the event, modifying the schedule state accordingly."""


def register_instruction(cls: type[SimulationEvent], instruction: str) -> None:
    """Register a SimulationEvent subclass as an instruction in the environment.

    To allow a SimulationEvent subclass to be used as an instruction in the
    environment, i.e. be part of the action space, it must be registered with
    a unique instruction name.

    Once registered, the instruction can be passed as part of an action during
    the `step` function and will be parsed into the corresponding SimulationEvent.

    Parameters
    ----------
    cls: type[SimulationEvent]
        The SimulationEvent subclass to be registered as an instruction.

    instruction: str
        The unique name of the instruction to be associated with the SimulationEvent subclass.

    Raises
    ------
    ValueError
        If the instruction name is already registered for a different SimulationEvent
        subclass.

    Example usage:
    >>> class MyEvent(SimulationEvent):
    ...     def __init__(self, arg1: int, arg2: str) -> None:
    ...         super().__init__()
    ...
    >>> register_instruction(MyEvent, "my_instruction")
    >>> env.step(("my_instruction", 42, "hello"))

    """
    if instruction in instructions:
        raise ValueError(
            f"Instruction '{instruction}' is already registered for {instructions[instruction]}"
        )

    instructions[instruction] = cls


EventID: TypeAlias = int
Rank: TypeAlias = int
PriorityValue: TypeAlias = float

# B-events (rank=-1) are processed before any C-event (rank>=0)
_TIMED_RANK: Rank = -1


class ScheduledEvent(EzPickle):
    """Record of an event scheduled for processing in the simulation."""

    event_id: EventID
    time: Time
    event: SimulationEvent

    rank: Rank
    priority: PriorityValue

    def __init__(
        self,
        event_id: EventID,
        time: Time,
        event: SimulationEvent,
        priority: PriorityValue,
        rank: Rank = _TIMED_RANK,
    ) -> None:
        self.event_id = event_id
        self.time = time
        self.event = event
        self.rank = rank
        self.priority = priority

    def __lt__(self, other: ScheduledEvent) -> bool:
        """Compare two ScheduledEvents for ordering in the event queue."""
        if self.time != other.time:
            return self.time < other.time

        if self.rank != other.rank:
            return self.rank < other.rank

        if self.priority != other.priority:
            return self.priority > other.priority

        return self.event_id < other.event_id

    def __eq__(self, other: object) -> bool:
        """Check if two ScheduledEvents are equal based on their event_id."""
        return (
            isinstance(other, ScheduledEvent)
            and self.event_id == other.event_id
        )

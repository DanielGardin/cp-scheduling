"""Base mechanism for the DES Backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeAlias

from cpscheduler.environment.backend.actions import Instruction
from cpscheduler.environment.constants import EzPickle, Time

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cpscheduler.environment.backend.des.des import DESBackend
    from cpscheduler.environment.state import ScheduleState


EventID: TypeAlias = int
Rank: TypeAlias = int
PriorityValue: TypeAlias = float

"""Implementation of events for the discrete event simulation environment."""


class SimulationEvent(Instruction["DESBackend"]):
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

    backend = "des"

    blocking: ClassVar[bool] = False

    # This are only used for C events
    def earliest_time(self, state: ScheduleState) -> Time | None:
        """Calculate the earliest time this event can be processed, given the current state.

        If None, the event is assumed to be always processable in the current time.
        """

    def is_ready(self, state: ScheduleState, backend: DESBackend) -> bool:
        """Check if the event is ready to be processed, given the current state."""
        return True


# B-events (rank=-1) are processed before any C-event (rank>=0)
_TIMED_RANK: Rank = -1


class ScheduledEvent(EzPickle):
    """Record of an event scheduled for processing in the simulation."""

    event_id: EventID
    time: Time
    event: SimulationEvent
    stale: bool

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
        self.stale = False

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

    def invalidate(self) -> None:
        """Invalidate the event, used when removing an event."""
        self.stale = True


class _EventQueue(EzPickle):
    events: list[ScheduledEvent]
    index_map: dict[EventID, int]

    def __init__(self) -> None:
        self.events = []
        self.index_map = {}

    def __bool__(self) -> bool:
        return bool(self.events)

    def __contains__(self, event_id: EventID) -> bool:
        return event_id in self.index_map

    def __iter__(self) -> Iterator[ScheduledEvent]:
        return iter(sorted(self.events))

    def __len__(self) -> int:
        return len(self.events)

    def get(self, event_id: EventID) -> ScheduledEvent:
        return self.events[self.index_map[event_id]]

    def push(self, entry: ScheduledEvent) -> None:
        pos = len(self.events)

        self.events.append(entry)
        self.index_map[entry.event_id] = pos

    def extend(self, entries: list[ScheduledEvent]) -> None:
        pos = len(self.events)

        self.events.extend(entries)
        for idx, entry in enumerate(entries, start=pos):
            self.index_map[entry.event_id] = idx

    def remove(self, event_id: EventID) -> None:
        if event_id not in self.index_map:
            raise KeyError(f"Event {event_id} not found in heap.")

        pos = self.index_map[event_id]
        last_pos = len(self.events) - 1

        removing_event = self.events[pos]
        removing_event.time = -1

        if pos != last_pos:
            last_event = self.events[last_pos]

            self.events[pos], self.events[last_pos] = last_event, removing_event
            self.index_map[last_event.event_id] = pos

        self.events.pop()
        del self.index_map[event_id]


class TimeSlot(EzPickle):
    """Helper class for managing events scheduled at a specific time step.

    Handles two heaps of events: one for timed events and one for non-timed events.
    """

    time: Time

    timed_events: _EventQueue
    non_timed_events: _EventQueue

    def __init__(self, time: Time) -> None:
        self.time = time
        self.timed_events = _EventQueue()
        self.non_timed_events = _EventQueue()

    def __contains__(self, event_id: EventID) -> bool:
        """Check whether an event ID is in this slot."""
        return (
            event_id in self.timed_events or event_id in self.non_timed_events
        )

    def __bool__(self) -> bool:
        """Return whether the time slot is empty or not."""
        return not self.timed_events and not self.non_timed_events

    def get_event(self, event_id: EventID) -> ScheduledEvent:
        """Return the event by its ID."""
        if event_id in self.timed_events:
            return self.timed_events.get(event_id)

        if event_id in self.non_timed_events:
            return self.non_timed_events.get(event_id)

        raise KeyError(f"Cannot get event {event_id} at time {self.time}.")

    def is_empty(self) -> bool:
        """Return whether this time slot has no events."""
        return bool(self)

    def add_timed_event(self, entry: ScheduledEvent) -> None:
        """Add a timed event to the time slot."""
        self.timed_events.push(entry)
        entry.time = self.time

    def add_non_timed_event(self, entry: ScheduledEvent) -> None:
        """Add a non timed event to the time slot."""
        self.non_timed_events.push(entry)
        entry.time = self.time

    def extend_non_timed_events(self, entries: list[ScheduledEvent]) -> None:
        """Add multiple non timed events to the time slot."""
        self.non_timed_events.extend(entries)
        for entry in entries:
            entry.time = self.time

    def remove_event(self, event_id: EventID) -> None:
        """Remove an event from its ID."""
        if event_id in self.timed_events:
            self.timed_events.remove(event_id)

        elif event_id in self.non_timed_events:
            self.non_timed_events.remove(event_id)

        else:
            raise KeyError(
                f"Event {event_id} not found in time slot {self.time}"
            )

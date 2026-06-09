"""Core components for the Discrete Event Simulation (DES) of the environment.

This module contains the main types and utilities for managing the event queue
and processing events according to their timing and blocking behavior.
"""

from __future__ import annotations

from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, ClassVar, TypeAlias

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle, Time

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cpscheduler.environment.state import ScheduleState

instructions: dict[str, type[SimulationEvent]] = {}

EventID = int

_global_event_id: EventID = 0


# FUTURE: The global event ID generator is a simple solution for ensuring
# unique event IDs, but it may not be ideal for all use cases, especially in
# multi-threaded or distributed environments.
# In the future, we may want to consider more robust solutions for generating
# unique event IDs, even better if they can be generated per environment.
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

    _event_id: EventID

    def __init__(self) -> None:
        """Initialize the SimulationEvent, assigning a unique event ID.

        The event ID is used for tracking and managing events in the schedule
        and is automatically assigned upon creation of the event.
        This ensures that each event can be uniquely identified and referenced
        throughout the simulation.
        """
        global _global_event_id

        self._event_id = _global_event_id
        _global_event_id += 1

    @property
    def event_id(self) -> EventID:
        """Unique identifier for this event, assigned upon creation."""
        return self._event_id

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


def _validate_event(
    event: SimulationEvent, state: ScheduleState
) -> SimulationEvent:
    while True:
        validated_event = event.resolve(state)

        if event is validated_event:
            return event

        event = validated_event


Rank: TypeAlias = int
PriorityValue: TypeAlias = int
OrderValue: TypeAlias = int

_Entry = tuple[Rank, PriorityValue, OrderValue, SimulationEvent]
_EntryKey = tuple[Rank, PriorityValue, OrderValue]


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class Schedule(EzPickle):
    """Discrete Event Schedule kernel for managing and processing events in the simulation.

    This class is responsible for maintaining the event queue, scheduling events
    according to their timing and blocking behavior, and providing an interface for
    adding, removing, and rescheduling events during the simulation.

    We use a three-phased approach (Pidd, 1998) to manage the event queue:
    1. Maintain a min-heap of scheduled times to determine the next time step.
    2. Process all timed events at the current time step, they must be ready to
    process, otherwise an error is raised.
    3. Process non-timed events at the current time step, sorted by their priority
    and order of insertion. If a non-timed event is not ready to be processed,
    it is deferred to a next time.
    Blocking events are a structural condition can cause the deferral of
    all subsequent non-timed events when they are not ready.
    """

    timed_events: dict[Time, list[SimulationEvent]]
    non_timed_events: dict[Time, list[_Entry]]

    _heap: list[Time]
    _event_cache: dict[EventID, SimulationEvent]
    _non_timed_event_keys: dict[EventID, _EntryKey]
    _event_time_cache: dict[EventID, Time]

    _tail: Time | None
    _next_order: OrderValue
    _next_rank: Rank

    def __init__(self) -> None:
        """Initialize the Schedule with empty event queues and reset state."""
        self.timed_events = {}
        self.non_timed_events = {}

        self._heap = []
        self._event_cache = {}
        self._non_timed_event_keys = {}
        self._event_time_cache = {}

        self._tail = None
        self._next_order = 0
        self._next_rank = 0

    def reset(self) -> None:
        """Reset the schedule to its initial empty state."""
        self.timed_events.clear()
        self.non_timed_events.clear()

        self._heap.clear()
        self._event_cache.clear()
        self._non_timed_event_keys.clear()
        self._event_time_cache.clear()

        self._tail = None
        self._next_order = 0
        self._next_rank = 0

    def is_empty(self) -> bool:
        """Check if there are no scheduled events."""
        for time in self._heap:
            if self.timed_events.get(time) or self.non_timed_events.get(time):
                return False

        return True

    def next_time(self) -> Time:
        """Get the next scheduled time for events."""
        return self._heap[0]

    def _create_time_slot(self, time: Time) -> None:
        """Create a new time slot for events, ensuring the heap is updated."""
        if time not in self.timed_events:
            self.timed_events[time] = []
            self.non_timed_events[time] = []
            heappush(self._heap, time)

    def _may_remove_time_slot(self, time: Time) -> None:
        """Remove a time slot if it has no more events, ensuring the heap is updated."""
        if self.timed_events.get(time):
            return
        if self.non_timed_events.get(time):
            return

        self.non_timed_events.pop(time, None)
        self.timed_events.pop(time, None)

        self._heap.remove(time)
        heapify(self._heap)

        if self._tail == time:
            self._tail = None

    def _reschedule_event(self, entry: _Entry, state: ScheduleState) -> None:
        event = entry[-1]

        time = event.earliest_time(state)
        current_time = state.time

        if time > current_time:
            self._create_time_slot(time)
            heappush(self.non_timed_events[time], entry)
            self._event_time_cache[event.event_id] = time

            if event.blocking and (self._tail is None or time > self._tail):
                self._tail = time

            return

        if time == current_time:
            # This guardrail is stronger than we need, it will block
            # feasible paths that use non-timed and timed events together
            raise RuntimeError(
                f"Event {event} is potentially deadlocking the event "
                "queue due to an action-dependent dependency that may "
                "never happen."
            )

        raise ValueError(
            f"Cannot reschedule events triggered by {event} to the past: "
            f"{time} < {state.time}."
        )

    def _reschedule_blocking_event(
        self, entries: list[_Entry], idx: int, state: ScheduleState
    ) -> None:
        """Reschedule non-timed events to the next time step, preserving their order."""
        if not entries:
            return

        first_event = entries[idx][-1]
        time = first_event.earliest_time(state)
        current_time = state.time

        if time == current_time:
            if not first_event.is_ready(state):
                # This guardrail is stronger than we need, it will block
                # feasible paths that use non-timed and timed events together
                raise RuntimeError(
                    f"Event {first_event} is potentially deadlocking the event "
                    "queue due to an action-dependent dependency that may "
                    "never happen."
                )

        elif time < current_time:
            raise ValueError(
                f"Cannot reschedule events triggered by {first_event} to the past: "
                f"{time} < {state.time}."
            )

        self._create_time_slot(time)
        for i in range(idx, len(entries)):
            entry = entries[i]
            event = entry[-1]

            heappush(self.non_timed_events[time], entry)
            self._event_time_cache[event.event_id] = time

        if first_event.blocking and (self._tail is None or time > self._tail):
            self._tail = time

    def instruction_queue(
        self, state: ScheduleState
    ) -> Iterator[SimulationEvent]:
        """Iterate over the events waiting to be processed at the current time step.

        This method implements the three-phased approach to processing events,
        yielding timed events first, followed by non-timed events, sorted by
        priority and order.
        """
        time = heappop(self._heap)

        assert time == state.time, (
            f"Next event time {time} does not match current state time {state.time}"
        )

        if time in self.timed_events:
            timed_events = self.timed_events[time]

            for event in timed_events:
                if not event.is_ready(state):
                    raise ValueError(
                        f"Event is not ready to be processed: {event} at time {time}"
                    )

                yield event

                event_id = event.event_id
                del self._event_cache[event_id]
                del self._event_time_cache[event_id]

            del self.timed_events[time]

        if time in self.non_timed_events:
            non_timed_events = self.non_timed_events[time]
            non_timed_events.sort()
            deferred_events: list[_Entry] = []

            for idx, entry in enumerate(non_timed_events):
                event = entry[-1]

                if not event.is_ready(state):
                    if event.blocking:
                        self._reschedule_blocking_event(
                            non_timed_events, idx, state
                        )
                        break

                    deferred_events.append(entry)
                    continue

                yield event

                event_id = event.event_id
                del self._event_cache[event_id]
                del self._event_time_cache[event_id]
                del self._non_timed_event_keys[event_id]

            for entry in deferred_events:
                self._reschedule_event(entry, state)

            del self.non_timed_events[time]

    # Public API
    # -----------
    # The following methods can be called by events during processing.

    def add_event(
        self,
        event: SimulationEvent,
        state: ScheduleState,
        time: Time | None = None,
        priority: int | None = None,
    ) -> None:
        """Add an event to the schedule.

        Parameters
        ----------
        event: SimulationEvent
            The event to be added to the schedule.

        state: ScheduleState
            The current state of the schedule, used for validating and scheduling the event.

        time: Time | None
            The time at which the event should be scheduled. If None, the event will be
            scheduled as a non-timed event at the current time step.

        priority: int | None
            The priority of the event, used for ordering non-timed events. Higher values
            indicate higher priority. This parameter is only applicable for non-timed events
            (when time is None) and will be ignored for timed events.

        Raises
        ------
        ValueError
            If the event is not valid for the current state, or if the time or priority
            parameters are invalid.

        """
        event = _validate_event(event, state)
        event_id = event.event_id

        if time is None:
            if priority is None:
                priority = 0

            order = self._next_order
            rank = self._next_rank

            self._next_order += 1
            if event.blocking:
                self._next_rank += 1

            time = state.time
            if self._tail is not None and time < self._tail:
                time = self._tail

            self._event_cache[event_id] = event
            self._event_time_cache[event_id] = time
            self._non_timed_event_keys[event_id] = (rank, -priority, order)

            self._create_time_slot(time)
            self.non_timed_events[time].append((rank, -priority, order, event))
            return

        if priority is not None:
            raise ValueError(
                "priority is only supported for non-timed (C) events"
            )

        if time < state.time:
            raise ValueError(
                f"Cannot schedule event in the past: {time} < {state.time}"
            )

        self._event_cache[event_id] = event
        self._event_time_cache[event_id] = time

        self._create_time_slot(time)
        self.timed_events[time].append(event)

    def remove_event(self, event_id: EventID) -> None:
        """Remove an event from the schedule, if it is still scheduled."""
        time = self._event_time_cache.pop(event_id)
        event = self._event_cache.pop(event_id)

        if event_id in self._non_timed_event_keys:
            rank, priority, order = self._non_timed_event_keys.pop(event_id)
            entry = (rank, priority, order, event)

            self.non_timed_events[time].remove(entry)
            heapify(self.non_timed_events[time])

        else:
            self.timed_events[time].remove(event)

        self._may_remove_time_slot(time)

    def reschedule_event(
        self, event_id: EventID, state: ScheduleState, new_time: Time
    ) -> None:
        """Reschedule an existing timed event to a new time."""
        if event_id not in self._event_cache:
            raise ValueError(f"Event {event_id} is not scheduled")

        if event_id in self._non_timed_event_keys:
            raise ValueError("Cannot reschedule non-timed events.")

        if new_time < state.time:
            raise ValueError(
                f"Cannot reschedule event to the past: {new_time} < {state.time}"
            )

        time = self._event_time_cache[event_id]

        if new_time == time:
            return

        event = self._event_cache[event_id]

        self.timed_events[time].remove(event)
        self._create_time_slot(new_time)
        self.timed_events[new_time].append(event)
        self._event_time_cache[event_id] = new_time

    def change_event_priority(
        self, event_id: EventID, new_priority: PriorityValue
    ) -> None:
        """Change the priority of an existing non-timed event."""
        if event_id not in self._event_cache:
            raise ValueError(
                f"change_event_priority: Event {event_id} is not scheduled"
            )

        if event_id not in self._non_timed_event_keys:
            event = self._event_cache[event_id]

            raise ValueError(
                "change_event_priority: Timed events do not handle priority."
            )

        time = self._event_time_cache[event_id]
        rank, old_priority, order = self._non_timed_event_keys[event_id]
        event = self._event_cache[event_id]

        old_entry = (rank, old_priority, order, event)
        new_entry = (rank, -new_priority, order, event)

        entries = self.non_timed_events[time]
        idx = entries.index(old_entry)

        entries[idx] = new_entry
        self._non_timed_event_keys[event_id] = (rank, -new_priority, order)
        heapify(entries)

    def clear_schedule(self) -> None:
        """Clear all scheduled events, resetting the schedule to an empty state."""
        self.reset()

    def peek_events(self) -> Iterator[SimulationEvent]:
        """Peek at all scheduled events in the order they would be processed, without modifying the schedule."""
        for time in sorted(self.timed_events):
            for event in self.timed_events[time]:
                yield event

        for time in sorted(self.non_timed_events):
            entries = sorted(self.non_timed_events[time])
            for *_, event in entries:
                yield event

    def peek_events_at_time(self, time: Time) -> Iterator[SimulationEvent]:
        """Peek at all scheduled events at a specific time, without modifying the schedule."""
        for event in self.timed_events.get(time, []):
            yield event

        entries = sorted(self.non_timed_events.get(time, []))
        for *_, event in entries:
            yield event

"""Discrete Event Schedule kernel for managing and processing events in the simulation."""

from __future__ import annotations

from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle, Time
from cpscheduler.environment.des._queue import EventQueue
from cpscheduler.environment.des.base import ScheduledEvent

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cpscheduler.environment.des.base import (
        EventID,
        PriorityValue,
        Rank,
        SimulationEvent,
    )
    from cpscheduler.environment.state import ScheduleState


def _validate_event(
    event: SimulationEvent, state: ScheduleState
) -> SimulationEvent:
    while True:
        validated_event = event.resolve(state)

        if event is validated_event:
            return event

        event = validated_event


class _TimeSlot:
    """Helper class for managing events scheduled at a specific time step.

    Handles two heaps of events: one for timed events and one for non-timed events.
    """

    time: Time

    timed_events: EventQueue
    non_timed_events: EventQueue

    def __init__(self, time: Time) -> None:
        self.time = time
        self.timed_events = EventQueue()
        self.non_timed_events = EventQueue()

    def __contains__(self, event_id: EventID) -> bool:
        return (
            event_id in self.timed_events or event_id in self.non_timed_events
        )

    def get_event(self, event_id: EventID) -> ScheduledEvent:
        if event_id in self.timed_events:
            return self.timed_events.get(event_id)

        if event_id in self.non_timed_events:
            return self.non_timed_events.get(event_id)

        raise KeyError(f"Cannot get event {event_id} at time {self.time}.")

    def is_empty(self) -> bool:
        return not self.timed_events and not self.non_timed_events

    def add_timed_event(self, entry: ScheduledEvent) -> None:
        self.timed_events.push(entry)
        entry.time = self.time

    def add_non_timed_event(self, entry: ScheduledEvent) -> None:
        self.non_timed_events.push(entry)
        entry.time = self.time

    def extend_non_timed_events(self, entries: list[ScheduledEvent]) -> None:
        self.non_timed_events.extend(entries)
        for entry in entries:
            entry.time = self.time

    def remove_event(self, event_id: EventID) -> None:
        if event_id in self.timed_events:
            self.timed_events.remove(event_id)

        elif event_id in self.non_timed_events:
            self.non_timed_events.remove(event_id)

        else:
            raise KeyError(
                f"Event {event_id} not found in time slot {self.time}"
            )


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

    _time_slots: dict[Time, _TimeSlot]
    _event_cache: dict[EventID, ScheduledEvent]

    _heap: list[Time]

    _next_event_id: EventID
    _next_rank: Rank
    _tail: Time | None

    def __init__(self) -> None:
        """Initialize the Schedule with empty event queues and reset state."""
        self._time_slots = {}
        self._event_cache = {}

        self._heap = []

        self._next_event_id = 0
        self._next_rank = 0
        self._tail = None

    def reset(self) -> None:
        """Reset the schedule to its initial empty state."""
        self._time_slots.clear()
        self._event_cache.clear()

        self._heap.clear()

        self._next_event_id = 0
        self._next_rank = 0
        self._tail = None

    def is_empty(self) -> bool:
        """Check if there are no scheduled events."""
        return not self._heap

    def next_time(self) -> Time:
        """Get the next scheduled time for events."""
        return self._heap[0]

    def _create_time_slot(self, time: Time) -> _TimeSlot:
        """Create a new time slot for events, ensuring the heap is updated."""
        if time not in self._time_slots:
            self._time_slots[time] = _TimeSlot(time)
            heappush(self._heap, time)

        return self._time_slots[time]

    def _may_remove_time_slot(self, time: Time) -> None:
        """Remove a time slot if it has no more events, ensuring the heap is updated."""
        if time in self._time_slots and self._time_slots[time].is_empty():
            del self._time_slots[time]

            self._heap.remove(time)
            heapify(self._heap)

            if self._tail == time:
                self._tail = None

    def _reschedule_event(
        self, entry: ScheduledEvent, state: ScheduleState
    ) -> None:
        event = entry.event

        time = event.earliest_time(state)
        current_time = state.time

        if time > current_time:
            time_slot = self._create_time_slot(time)
            time_slot.add_non_timed_event(entry)

            if event.blocking and (self._tail is None or time > self._tail):
                self._tail = time

        elif time == current_time:
            # This guardrail is stronger than we need, it will block
            # feasible paths that use non-timed and timed events together
            raise RuntimeError(
                f"Event {event} is potentially deadlocking the event "
                "queue due to an action-dependent dependency that may "
                "never happen."
            )

        else:
            raise ValueError(
                f"Cannot reschedule events triggered by {event} to the past: "
                f"{time} < {state.time}."
            )

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

        time_slot = self._time_slots.pop(time)
        time_cache = self._event_cache

        for entry in time_slot.timed_events:
            event = entry.event

            if not event.is_ready(state):
                raise ValueError(
                    f"Event is not ready to be processed: {event} at time {time}"
                )

            yield event

            del time_cache[entry.event_id]

        deferred_events: list[ScheduledEvent] = []
        for entry in time_slot.non_timed_events:
            event = entry.event

            if event.is_ready(state):
                yield event
                time_slot.remove_event(entry.event_id)
                del time_cache[entry.event_id]
                continue

            if event.blocking:
                next_time = event.earliest_time(state)

                if self._tail is None or next_time > self._tail:
                    self._tail = next_time

                if next_time > time:
                    next_time_slot = self._create_time_slot(next_time)

                    next_time_slot.extend_non_timed_events(
                        time_slot.non_timed_events.events
                    )

                elif next_time == time:
                    raise RuntimeError(
                        f"Event {event} is potentially deadlocking the event "
                        "queue due to an action-dependent dependency that may "
                        "never happen."
                    )

                else:
                    raise ValueError(
                        f"Cannot reschedule events triggered by {event} to the past: "
                        f"{time} < {state.time}."
                    )

                break

            deferred_events.append(entry)

        for entry in deferred_events:
            self._reschedule_event(entry, state)

    # Public API
    # -----------
    # The following methods can be called by events during processing.

    def add_event(
        self,
        event: SimulationEvent,
        state: ScheduleState,
        time: Time | None = None,
        priority: float | None = None,
    ) -> EventID:
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

        priority: float | None
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
        event_id = self._next_event_id
        priority = priority if priority is not None else 0.0

        if time is None:
            rank = self._next_rank
            if event.blocking:
                self._next_rank += 1

            time = state.time
            if self._tail is not None and time < self._tail:
                time = self._tail

            entry = ScheduledEvent(event_id, time, event, priority, rank)

            time_slot = self._create_time_slot(time)
            time_slot.add_non_timed_event(entry)

        elif time < state.time:
            raise ValueError(
                f"Cannot schedule event in the past: {time} < {state.time}"
            )

        else:
            entry = ScheduledEvent(event_id, time, event, priority)

            time_slot = self._create_time_slot(time)
            time_slot.add_timed_event(entry)

        self._event_cache[event_id] = entry
        self._next_event_id += 1
        return event_id

    def remove_event(self, event_id: EventID) -> None:
        """Remove an event from the schedule, if it is still scheduled."""
        if event_id not in self._event_cache:
            raise KeyError(f"Event {event_id} is not scheduled")

        time = self._event_cache.pop(event_id).time
        self._time_slots[time].remove_event(event_id)
        self._may_remove_time_slot(time)

    def reschedule_event(
        self, event_id: EventID, state: ScheduleState, new_time: Time
    ) -> None:
        """Reschedule an existing timed event to a new time."""
        if event_id not in self._event_cache:
            raise ValueError(f"Event {event_id} is not scheduled")

        entry = self._event_cache[event_id]
        old_time = entry.time

        time_slot = self._time_slots[old_time]

        if event_id in time_slot.non_timed_events:
            raise ValueError("Cannot reschedule non-timed events.")

        if old_time < new_time:
            time_slot.remove_event(event_id)

            new_time_slot = self._create_time_slot(new_time)
            new_time_slot.add_timed_event(entry)

            self._may_remove_time_slot(old_time)

        elif new_time < state.time:
            raise ValueError(
                f"Cannot reschedule event to the past: {new_time} < {state.time}"
            )

    def change_event_priority(
        self, event_id: EventID, new_priority: PriorityValue
    ) -> None:
        """Change the priority of an existing non-timed event."""
        if event_id not in self._event_cache:
            raise ValueError(f"Event {event_id} is not scheduled")

        entry = self._event_cache[event_id]
        entry.priority = new_priority

    def clear_schedule(self) -> None:
        """Clear all scheduled events, resetting the schedule to an empty state."""
        self.reset()

    def peek_events(self) -> Iterator[SimulationEvent]:
        """Peek at all scheduled events in the order they would be processed, without modifying the schedule."""
        for entry in sorted(self._event_cache.values()):
            yield entry.event

    def peek_events_at_time(self, time: Time) -> Iterator[SimulationEvent]:
        """Peek at all scheduled events at a specific time, without modifying the schedule."""
        if time not in self._time_slots:
            return

        time_slot = self._time_slots[time]

        for entry in time_slot.timed_events:
            yield entry.event

        for entry in time_slot.non_timed_events:
            yield entry.event

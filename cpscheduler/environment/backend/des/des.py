"""DES Backend class."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.backend.backend import ScheduleBackend
from cpscheduler.environment.backend.des.base import (
    EventID,
    PriorityValue,
    Rank,
    ScheduledEvent,
    SimulationEvent,
    TimeSlot,
)
from cpscheduler.environment.constants import MAX_TIME, TaskID, Time

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cpscheduler.environment.backend import Instruction
    from cpscheduler.environment.state import ScheduleState


TIMED_STAGE = 0
NON_TIMED_STAGE = 1
TIME_ADVANCE_STAGE = 2


# FUTURE: There are a lot of changes I want to make in this backend that I
# cannot implement in the current version (0.9.0) due to other higher priority changes.
# Here is a list of changes that I think may enhance this backend:
#   1. A better way of detecting whether an instruction is deadlocked is by
# implementing a method alongside `earliest_start`, like `is_unlocked`, which
# hints to the backend whether the time in the earliest start is exact, or an heuristic.
#
#   2. There is no way of a ready instruction to block other non-timed instructions.
# This behavior is valuable for the case where we introduce the NOOP instruction.
# Instead of trying to execute a task now, the NOOP jumps to the next decision point.
# This is the mechanism in some RL environmnts to avoid a forced non-delay generation.
@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class DESBackend(ScheduleBackend):
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

    backend = "des"

    _time_slots: dict[Time, TimeSlot]
    _event_cache: dict[EventID, ScheduledEvent]

    _heap: list[Time]

    _next_event_id: EventID
    _next_rank: Rank
    _tail: Time | None

    time: Time
    stage: int
    instruction_idx: int

    # Explicit dispatch state for the time slot currently being processed.
    _current_time_slot: TimeSlot | None
    _timed_snapshot: list[ScheduledEvent]
    _non_timed_snapshot: list[ScheduledEvent]
    _timed_index: int
    _non_timed_index: int
    _deferred_events: list[ScheduledEvent]

    def __init__(self) -> None:
        """Initialize the Schedule with empty event queues and reset state."""
        self._time_slots = {}
        self._event_cache = {}

        self._heap = []

        self._next_event_id = 0
        self._next_rank = 0
        self._tail = None

        self.time = 0
        self.stage = TIMED_STAGE
        self.instruction_idx = 0

        self._current_time_slot = None
        self._timed_snapshot = []
        self._non_timed_snapshot = []
        self._timed_index = 0
        self._non_timed_index = 0
        self._deferred_events = []

    def reset(self) -> None:
        """Reset the schedule to its initial empty state."""
        self._time_slots.clear()
        self._event_cache.clear()

        self._heap.clear()

        self._next_event_id = 0
        self._next_rank = 0
        self._tail = None

        self.time = 0
        self.stage = TIMED_STAGE
        self.instruction_idx = 0

        self._current_time_slot = None
        self._timed_snapshot = []
        self._non_timed_snapshot = []
        self._timed_index = 0
        self._non_timed_index = 0
        self._deferred_events = []

    @override
    def get_eligible_set(self, state: ScheduleState) -> list[TaskID]:
        return state.get_available_tasks(self.time)

    def is_empty(self) -> bool:
        """Check if there are no scheduled events."""
        return not self._heap

    def _create_time_slot(self, time: Time) -> TimeSlot:
        """Create a new time slot for events, ensuring the heap is updated."""
        if time not in self._time_slots:
            self._time_slots[time] = TimeSlot(time)
            heappush(self._heap, time)

        return self._time_slots[time]

    def dispatch_instruction(
        self, state: ScheduleState
    ) -> SimulationEvent | None:
        """Dispatch the next event by consuming an internal generator."""
        while self._heap:
            if self.stage == TIMED_STAGE:
                event = self._step_timed_stage(state)

            elif self.stage == NON_TIMED_STAGE:
                event = self._step_non_timed_stage(state)

            else:
                raise RuntimeError(f"Invalid dispatch stage: {self.stage}")

            if event is not None:
                return event

        self.stage = TIME_ADVANCE_STAGE

        next_time = max(self.time, state.get_earliest_start_lb())
        self.time = (
            next_time if next_time < MAX_TIME else state.get_latest_end()
        )

        self.stage = TIMED_STAGE
        return None

    def _step_timed_stage(self, state: ScheduleState) -> SimulationEvent | None:
        """Process one timed event of the current time slot, or advance to the non-timed stage."""
        if self._current_time_slot is None:
            time = self._heap[0]
            self.time = time

            self._current_time_slot = self._time_slots.pop(time)
            self._timed_snapshot = list(self._current_time_slot.timed_events)
            self._timed_index = 0

        if self._timed_index < len(self._timed_snapshot):
            entry = self._timed_snapshot[self._timed_index]
            event = entry.event

            if not event.is_ready(state, self):
                raise ValueError(
                    f"Event is not ready to be processed: {event} at time {self.time}"
                )

            self._timed_index += 1
            del self._event_cache[entry.event_id]

            return event

        self.stage = NON_TIMED_STAGE
        self._non_timed_snapshot = list(
            self._current_time_slot.non_timed_events
        )
        self._non_timed_index = 0
        self._deferred_events.clear()

        return None

    def _step_non_timed_stage(
        self, state: ScheduleState
    ) -> SimulationEvent | None:
        """Process one non-timed event of the current time slot, or finalize the slot."""
        assert self._current_time_slot is not None
        slot = self._current_time_slot

        if self._non_timed_index < len(self._non_timed_snapshot):
            entry = self._non_timed_snapshot[self._non_timed_index]
            event = entry.event

            if event.is_ready(state, self):
                self._non_timed_index += 1
                slot.remove_event(entry.event_id)
                del self._event_cache[entry.event_id]

                return event

            if event.blocking:
                next_time = event.earliest_time(state)
                self._defer_blocking_event(event, next_time)
                self._finalize_time_slot()

                return None

            self._deferred_events.append(entry)
            self._non_timed_index += 1
            return None

        for entry in self._deferred_events:
            self._reschedule_event(entry, state)

        self._finalize_time_slot()
        return None

    def _defer_blocking_event(
        self, event: SimulationEvent, next_time: Time | None
    ) -> None:
        """Handle a blocking, not-ready non-timed event: sweep remaining entries forward and end the slot."""
        assert self._current_time_slot is not None
        time = self.time

        if next_time is None or next_time <= self.time:
            raise RuntimeError(
                f"Event {event} is potentially deadlocking the event "
                "queue: It is not ready, but its earliest time is earlier than "
                f"the current time ({next_time} <= {self.time})"
            )

        if next_time > time:
            next_time_slot = self._create_time_slot(next_time)

            next_time_slot.extend_non_timed_events(
                self._current_time_slot.non_timed_events.events
            )

        if self._tail is None or next_time > self._tail:
            self._tail = next_time

    def _finalize_time_slot(self) -> None:
        """Close out the current time slot and pop it from the heap."""
        heappop(self._heap)

        self._current_time_slot = None
        self._timed_snapshot.clear()
        self._non_timed_snapshot.clear()
        self._timed_index = 0
        self._non_timed_index = 0
        self._deferred_events.clear()
        self.stage = TIMED_STAGE

    def _reschedule_event(
        self, entry: ScheduledEvent, state: ScheduleState
    ) -> None:
        event = entry.event

        time = event.earliest_time(state)

        if time is None or time <= self.time:
            raise RuntimeError(
                f"Event {event} is potentially deadlocking the event "
                "queue: It is not ready, but its earliest time is earlier than "
                f"the current time ({time} <= {self.time})"
            )

        if time > self.time:
            time_slot = self._create_time_slot(time)
            time_slot.add_non_timed_event(entry)

            if event.blocking and (self._tail is None or time > self._tail):
                self._tail = time

        else:
            raise ValueError(
                f"Cannot reschedule events triggered by {event} to the past: "
                f"{time} < {self.time}."
            )

    @override
    def get_info(self) -> dict[str, Any]:
        return {"current_time": self.time}

    # Public API
    # -----------
    # The following methods can be called by events during processing.

    @override
    def add_instruction(
        self,
        instruction: Instruction[DESBackend],
        time: Time | None = None,
        priority: float | None = None,
    ) -> EventID:
        """Add an event to the schedule.

        Parameters
        ----------
        instruction: SimulationEvent
            The event to be added to the schedule.

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
            If the instruction is not a DES instruction.
            If the event is not valid for the current state, or if the time or priority
            parameters are invalid.

        """
        if not isinstance(instruction, SimulationEvent):
            raise ValueError(
                f"{type(instruction).__name__} is not a subclass of SimulationEvent "
                f"DES instructions are expected to subclass SimulationEvent."
            )

        event_id = super().add_instruction(instruction, time, priority)
        priority = priority if priority is not None else 0.0

        if time is None:
            rank = self._next_rank
            if instruction.blocking:
                self._next_rank += 1

            time = self.time
            if self._tail is not None and time < self._tail:
                time = self._tail

            entry = ScheduledEvent(event_id, time, instruction, priority, rank)

            time_slot = self._create_time_slot(time)
            time_slot.add_non_timed_event(entry)

        elif time < self.time:
            raise ValueError(
                f"Cannot schedule event in the past: {time} < {self.time}"
            )

        else:
            entry = ScheduledEvent(event_id, time, instruction, priority)

            time_slot = self._create_time_slot(time)
            time_slot.add_timed_event(entry)

        self._event_cache[event_id] = entry
        return event_id

    def remove_event(self, event_id: EventID) -> None:
        """Remove an event from the schedule, if it is still scheduled."""
        if event_id not in self._event_cache:
            raise KeyError(f"Event {event_id} is not scheduled")

        time = self._event_cache.pop(event_id).time
        self._time_slots[time].remove_event(event_id)

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

        if new_time < self.time:
            raise ValueError(
                f"Cannot reschedule event to the past: {new_time} < {self.time}"
            )

        time_slot.remove_event(event_id)

        new_time_slot = self._create_time_slot(new_time)
        new_time_slot.add_timed_event(entry)

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

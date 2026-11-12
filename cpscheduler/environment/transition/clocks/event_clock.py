"""Event clock, a discrete event simulation of the instruction queue."""

from __future__ import annotations

from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, Any

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import MAX_TIME, Time
from cpscheduler.environment.mixins import EzPickle
from cpscheduler.environment.transition.actions import TimeUpdateInstruction
from cpscheduler.environment.transition.core import GlobalClock

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.actions import Instruction
    from cpscheduler.environment.transition.core import Transition

EventID = int
PriorityValue = float
Rank = int

TIMED_STAGE = 0
NON_TIMED_STAGE = 1

# B-events (rank=-1) are processed before any C-event (rank>=0)
_TIMED_RANK: Rank = -1


class ScheduledEvent(EzPickle):
    """Record of an event scheduled for processing in the simulation."""

    event_id: EventID
    time: Time
    event: Instruction[Any]
    stale: bool

    rank: Rank
    priority: PriorityValue

    def __init__(
        self,
        event_id: EventID,
        time: Time,
        event: Instruction[Any],
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
        return bool(self.timed_events) or bool(self.non_timed_events)

    def get_event(self, event_id: EventID) -> ScheduledEvent:
        """Return the event by its ID."""
        if event_id in self.timed_events:
            return self.timed_events.get(event_id)

        if event_id in self.non_timed_events:
            return self.non_timed_events.get(event_id)

        raise KeyError(f"Cannot get event {event_id} at time {self.time}.")

    def is_empty(self) -> bool:
        """Return whether this time slot has no events."""
        return not self

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


# FUTURE: A better way of detecting whether an instruction is deadlocked is by
# implementing a method alongside `earliest_time`, like `is_unlocked`, which
# hints to the clock whether the earliest time is exact, or an heuristic.
#
# FUTURE: This clock has a lot of error handling, which is good for debugging,
# but it is bad for RL training when an infeasible action is not masked.
# An idea is to implement an explicit parameter to tell how to handle errors,
# for instance: raise, terminate, truncate, ignore...
@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class EventClock(GlobalClock):
    """A clock that jumps to the next time at which something can happen.

    We use a three-phased approach (Pidd, 1998) to manage the queue:
    1. Maintain a min-heap of scheduled times to determine the next time.
    2. Process all timed instructions of the current time, they must be ready,
    otherwise an error is raised.
    3. Process the untimed instructions, sorted by their priority and order of
    insertion. One that is not ready is deferred to its earliest time, and a
    blocking one defers every instruction queued after it too.

    When the queue is empty, the clock jumps to the earliest time at which a
    task can start, so the next decision always has a candidate. This is what
    makes the schedules non-delay when one task is executed per step.
    """

    _time_slots: dict[Time, TimeSlot]
    _event_cache: dict[EventID, ScheduledEvent]

    _heap: list[Time]

    _next_rank: Rank
    _tail: Time | None

    stage: int

    # Explicit dispatch state for the time slot currently being processed.
    _current_time_slot: TimeSlot | None
    _timed_snapshot: list[ScheduledEvent]
    _non_timed_snapshot: list[ScheduledEvent]
    _timed_index: int
    _non_timed_index: int
    _deferred_events: list[ScheduledEvent]

    # Queue control variables
    _explicit_defer_time: Time
    _explicit_halt: bool

    # Last clock value announced to time-aware constraints
    _announced_time: Time

    # Backtracking
    _time_trail: list[Time]
    _tail_trail: list[Time | None]

    def __init__(self) -> None:
        """Initialize the Schedule with empty event queues and reset state."""
        self._time_slots = {}
        self._event_cache = {}

        self._heap = []

        self._next_rank = 0
        self._tail = None

        self.time = 0
        self.stage = TIMED_STAGE

        self._current_time_slot = None
        self._timed_snapshot = []
        self._non_timed_snapshot = []
        self._timed_index = 0
        self._non_timed_index = 0
        self._deferred_events = []

        self._explicit_defer_time = 0
        self._explicit_halt = False

        self._announced_time = 0

        self._time_trail = []
        self._tail_trail = []

    def _drop_pending_events(self) -> None:
        """Remove every scheduled event, without moving the clock."""
        self._time_slots.clear()
        self._event_cache.clear()

        self._heap.clear()

        self.stage = TIMED_STAGE

        self._current_time_slot = None
        self._timed_snapshot.clear()
        self._non_timed_snapshot.clear()
        self._timed_index = 0
        self._non_timed_index = 0
        self._deferred_events.clear()

        self._explicit_defer_time = 0
        self._explicit_halt = False

    @override
    def reset(self) -> None:
        self._drop_pending_events()

        self._next_rank = 0
        self._tail = None

        self.time = 0
        self._announced_time = 0

        self._time_trail.clear()
        self._tail_trail.clear()

    @override
    @override
    def start_episode(self, state: ScheduleState) -> None:
        # Same behavior of the termination phase: the first decision point is
        # where some task can start, so the initial action support is not empty.
        earliest_start = state.get_earliest_start_lb()

        if earliest_start != MAX_TIME:
            self.time = max(self.time, earliest_start)
            self._announced_time = self.time

    @override
    def is_empty(self) -> bool:
        return not self._heap

    def _create_time_slot(self, time: Time) -> TimeSlot:
        """Create a new time slot for events, ensuring the heap is updated."""
        if time not in self._time_slots:
            self._time_slots[time] = TimeSlot(time)
            heappush(self._heap, time)

        return self._time_slots[time]

    @override
    def pop(
        self, state: ScheduleState, transition: Transition[Any]
    ) -> Instruction[Any] | None:
        if self._explicit_halt:
            self._explicit_halt = False
            return None

        while self._heap:
            # Stage A: Advance time to the earliest event
            if self._current_time_slot is None:
                time = self._heap[0]
                self.time = time

                current_slot = self._time_slots.pop(time)

                self._current_time_slot = current_slot
                self._timed_snapshot = list(current_slot.timed_events)
                self._timed_index = 0

                # The constraints must see the new time before any event
                # readiness is evaluated for it.
                if transition.tracks_time and time != self._announced_time:
                    self._announced_time = time
                    return TimeUpdateInstruction(time)

            event = None
            # Stage B: Run all unconditional events
            if self.stage == TIMED_STAGE:
                event = self._step_timed_stage(state, transition)

            # Stage C: Check conditional events
            if self.stage == NON_TIMED_STAGE:
                event = self._step_non_timed_stage(state, transition)

            if event is not None:
                return event

            self._finalize_time_slot(state)

        # Termination Phase: Advance to a potential next event
        # This is the behavior that make DES non-delay by default, the termination
        # phase always jumps to a state where |eligible_tasks| > 0, or to
        # the latest end, when such state does not exist.
        next_time = (
            state.get_latest_end()
            if state.is_terminal()
            else state.get_earliest_start_lb()
        )

        # No task can start, not even in the future. Time is not advanced to
        # avoid leaking the MAX_TIME sentinel.
        if next_time != MAX_TIME:
            self.time = max(self.time, next_time)

        self.stage = TIMED_STAGE

        if not state.is_terminal() and state.expire_tasks(self.time):
            return TimeUpdateInstruction(self.time)

        if transition.tracks_time and self.time != self._announced_time:
            self._announced_time = self.time
            return TimeUpdateInstruction(self.time)

        return None

    def _step_timed_stage(
        self, state: ScheduleState, transition: Transition[Any]
    ) -> Instruction[Any] | None:
        """Process one timed event of the current time slot, or advance to the non-timed stage."""
        assert self._current_time_slot is not None

        idx = self._timed_index
        snapshot = self._timed_snapshot
        size = len(snapshot)

        while idx < size:
            entry = snapshot[idx]
            idx += 1

            if entry.stale:
                continue

            event = entry.event

            if not event.is_ready(state, transition):
                raise ValueError(
                    f"Event is not ready to be processed: {event} at time {self.time}"
                )

            del self._event_cache[entry.event_id]
            self._timed_index = idx

            return event

        self._timed_index = idx
        self.stage = NON_TIMED_STAGE
        self._non_timed_snapshot = list(
            self._current_time_slot.non_timed_events
        )
        self._non_timed_index = 0
        self._deferred_events.clear()

        return None

    def _step_non_timed_stage(
        self, state: ScheduleState, transition: Transition[Any]
    ) -> Instruction[Any] | None:
        """Process one non-timed event of the current time slot, or finalize the slot."""
        assert self._current_time_slot is not None

        if self._explicit_defer_time:
            self._defer_remaining_events(self._explicit_defer_time)

            # Not sure if we really need this guy, I imagine so because it
            # triggers `tight_global_time` in the state, which is only required
            # when the schedule is non-delay (conjecture)
            transition.add_instruction(
                TimeUpdateInstruction(self._explicit_defer_time),
                self._explicit_defer_time,
            )
            self._explicit_defer_time = 0

            return None

        idx = self._non_timed_index
        snapshot = self._non_timed_snapshot
        size = len(snapshot)

        while idx < size:
            entry = snapshot[idx]
            idx += 1

            if entry.stale:
                self._remove_event(entry.event_id)
                continue

            event = entry.event

            if event.is_ready(state, transition):
                self._remove_event(entry.event_id)

                self._non_timed_index = idx
                return event

            if event.blocking:
                next_time = event.earliest_time(state)

                if next_time is None:
                    raise RuntimeError(
                        f"Event {event} is locking the event queue without a "
                        "earliest time to be ready. It might require another "
                        "instruction to be processed first, or it might be deadlocked."
                    )

                self._defer_remaining_events(next_time)

                return None

            self._deferred_events.append(entry)

        return None

    def _remove_event(self, event_id: EventID) -> None:
        assert self._current_time_slot is not None

        self._current_time_slot.non_timed_events.remove(event_id)

        # Stale events have already left the cache when they were invalidated
        self._event_cache.pop(event_id, None)

    def _defer_remaining_events(self, next_time: Time) -> None:
        """Defer all remaining C-events due to a blocking signal to `next time`."""
        assert self._current_time_slot is not None
        current_time = self.time

        if next_time <= current_time:
            raise RuntimeError(
                "Cannot defer C-events to a time earlier than the current time "
                f"({next_time} <= {self.time})."
            )

        # Events that were individually deferred are rescheduled at the end of
        # this time slot, moving them here too would duplicate them.
        individually_deferred = {
            entry.event_id for entry in self._deferred_events
        }

        next_time_slot = self._create_time_slot(next_time)
        next_time_slot.extend_non_timed_events(
            [
                entry
                for entry in self._current_time_slot.non_timed_events.events
                if entry.event_id not in individually_deferred
            ]
        )

        if self._tail is None or next_time > self._tail:
            self._tail = next_time

    def _finalize_time_slot(self, state: ScheduleState) -> None:
        """Close out the current time slot and pop it from the heap."""
        heappop(self._heap)

        for entry in self._deferred_events:
            self._reschedule_event(entry, state)

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
    # Public API
    # -----------
    # The following methods can be called by instructions during processing.

    @override
    def push(
        self,
        instruction: Instruction[Any],
        time: Time | None,
        priority: float | None,
        event_id: int,
    ) -> None:
        """Queue an instruction.

        A timed instruction is processed exactly at its time, and it must be
        ready then. An untimed one is processed at the current time, by
        highest priority, then in insertion order, as soon as it is ready.
        """
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

    def remove_event(self, event_id: EventID) -> None:
        """Remove an event from the schedule, if it is still scheduled."""
        if event_id not in self._event_cache:
            raise KeyError(f"Event {event_id} is not scheduled")

        entry = self._event_cache.pop(event_id)
        time = entry.time

        if time == self.time:
            # We cannot simply remove it, as it is already cached.
            entry.invalidate()

        else:
            self._time_slots[time].remove_event(event_id)
            self._discard_if_empty(time)

    def _discard_if_empty(self, time: Time) -> None:
        """Drop an empty future time slot, so the clock does not visit it."""
        if self._time_slots[time].is_empty():
            del self._time_slots[time]
            self._heap.remove(time)
            heapify(self._heap)

    def reschedule_event(self, event_id: EventID, new_time: Time) -> None:
        """Reschedule an existing timed event to a new time."""
        if event_id not in self._event_cache:
            raise ValueError(f"Event {event_id} is not scheduled")

        entry = self._event_cache[event_id]
        old_time = entry.time

        time_slot = self._time_slots.get(old_time)

        if time_slot is None:
            raise ValueError(
                f"Event {event_id} is being processed, it cannot be rescheduled."
            )

        if event_id in time_slot.non_timed_events:
            raise ValueError("Cannot reschedule non-timed events.")

        if new_time < self.time:
            raise ValueError(
                f"Cannot reschedule event to the past: {new_time} < {self.time}"
            )

        time_slot.remove_event(event_id)

        new_time_slot = self._create_time_slot(new_time)
        new_time_slot.add_timed_event(entry)

        self._discard_if_empty(old_time)

    def change_event_priority(
        self, event_id: EventID, new_priority: PriorityValue
    ) -> None:
        """Change the priority of an existing non-timed event."""
        if event_id not in self._event_cache:
            raise ValueError(f"Event {event_id} is not scheduled")

        entry = self._event_cache[event_id]
        entry.priority = new_priority

    def peek_events(self) -> Iterator[Instruction[Any]]:
        """Peek at all scheduled events in the order they would be processed, without modifying the schedule."""
        for entry in sorted(self._event_cache.values()):
            yield entry.event

    def peek_events_at_time(self, time: Time) -> Iterator[Instruction[Any]]:
        """Peek at all scheduled events at a specific time, without modifying the schedule."""
        if time not in self._time_slots:
            return

        time_slot = self._time_slots[time]

        for entry in time_slot.timed_events:
            yield entry.event

        for entry in time_slot.non_timed_events:
            yield entry.event

    @override
    def halt(self) -> None:
        self._explicit_halt = True

    # FUTURE: The interaction between non-blocking events and advancement is
    # not entirely known. We should make tests that stress that interface.
    @override
    def advance_to(self, time: Time) -> None:
        # The clock is not moved here, as this would skip a timed event that
        # is due before `time`. The untimed events are deferred to `time`
        # instead, which gives a proper jump when no timed event exists.
        if time < self.time:
            raise RuntimeError(
                "Cannot advance to a time earlier than the current time "
                f"({time} <= {self.time})."
            )

        self._explicit_defer_time = time

    @override
    def checkpoint(self, mark: int) -> None:
        assert len(self._time_trail) == mark

        self._time_trail.append(self.time)
        self._tail_trail.append(self._tail)

    @override
    def backtrack(self, mark: int) -> None:
        # Checkpoints are only taken with an empty queue, but a step that
        # fails stops with its remaining instructions still scheduled.
        self._drop_pending_events()

        self.time = self._time_trail[mark]
        self._announced_time = self.time
        self._tail = self._tail_trail[mark]

        del self._time_trail[mark:]
        del self._tail_trail[mark:]


# Why `advance_to`, and in which scenario it is used?
#
# The deferring process in the DES kernel can only happen for an event that
# is both _Blocking_ and _Not Ready_, which indicates that only time, or
# other instruction can unblock the queue and let other instruction in the
# C-event queue to run. The potential earliest time which this can happen is
# the implementation of `earliest_time`.
#
# What does it not cover? When an event want to defer other instructions, but
# want to be removed from the queue.
# This behavior cannot happen in the scenario above, because not being ready is
# a requirement for deferring. `advance_to` fills this specific gap, it is removed
# when ready, but can tell the simulator to defer other instructions to an
# "earliest time" given as a parameter (`advance_to(time)`).

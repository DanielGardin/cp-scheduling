from typing import Any, ClassVar, TypeAlias
from collections.abc import Iterator

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import Time
from cpscheduler.environment.state import ScheduleState

from heapq import heappush, heappop, heapify

instructions: dict[str, type["SimulationEvent"]] = {}


@mypyc_attr(allow_interpreted_subclasses=True)
class SimulationEvent:
    """
    Base class for all events in the simulation.

    Events are result from processing instructions or from the environment
    itself, triggering changes in the schedule state.

    To create a new event, subclass this class to define the instruction type
    and behavior in the simulation.
    """

    blocking: ClassVar[bool] = False
    "Whether this event blocks the processing of subsequent events."

    def __init__(self, *args: Any) -> None:
        pass

    @property
    def args(self) -> tuple[int, ...]:
        "The arguments of the event, used for logging and instruction generation."
        raise NotImplementedError(
            "Subclasses must implement the args property."
        )

    def __repr__(self) -> str:
        args = ", ".join(str(arg) for arg in self.args)
        return f"{self.__class__.__name__}({args})"

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (self.__class__, self.args, ())

    def __hash__(self) -> int:
        return id(self)

    def resolve(self, state: ScheduleState) -> None:
        "Resolve and statically validate the event."

    # This are only used for C events
    def earliest_time(self, state: ScheduleState) -> Time:
        "Calculate the earliest time this event can be processed, given the current state."
        return state.time

    def is_ready(self, state: ScheduleState) -> bool:
        "Check if the event is ready to be processed, given the current state."
        return True

    def process(self, state: ScheduleState, schedule: "Schedule") -> None:
        "Process the event, modifying the schedule state accordingly."


def register_instruction(cls: type[SimulationEvent], instruction: str) -> None:
    """
    To allow a SimulationEvent subclass to be used as an instruction in the
    environment, i.e. be part of the action space, it must be registered with
    a unique instruction name.

    Once registered, the instruction can be passed as part of an action during
    the `step` function and will be parsed into the corresponding SimulationEvent.

    Example usage:
    ```python
    class MyEvent(SimulationEvent):
        def __init__(self, arg1: int, arg2: str) -> None:
            ...

    register_instruction(MyEvent, "my_instruction")

    env.step(("my_instruction", 42, "hello"))
    ```
    """
    if instruction in instructions:
        raise ValueError(
            f"Instruction '{instruction}' is already registered for {instructions[instruction]}"
        )

    instructions[instruction] = cls


Rank: TypeAlias = int
PriorityValue: TypeAlias = int
OrderValue: TypeAlias = int

_Entry = tuple[Rank, PriorityValue, OrderValue, SimulationEvent]
_EntryKey = tuple[Rank, PriorityValue, OrderValue]


class Schedule:
    """
    The main kernel of the DES environment, responsible for managing the
    event queue and processing events according to their timing and blocking
    behavior.
    """

    __slots__ = (
        "_heap",
        "_timed_events",
        "_non_timed_events",
        "_event_keys",
        "_event_earliest_time_cache",
        "_tail",
        "_next_order",
        "_next_rank",
    )

    _heap: list[Time]
    _timed_events: dict[Time, list[SimulationEvent]]
    _non_timed_events: dict[Time, list[_Entry]]

    _event_keys: dict[SimulationEvent, _EntryKey]
    _event_earliest_time_cache: dict[SimulationEvent, Time]

    _tail: Time | None
    _next_order: OrderValue
    _next_rank: Rank

    def __init__(self) -> None:
        self._heap = []
        self._timed_events = {}
        self._non_timed_events = {}

        self._event_keys = {}
        self._event_earliest_time_cache = {}

        self._tail = None
        self._next_order = 0
        self._next_rank = 0

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            (),
            (
                self._heap,
                self._timed_events,
                self._non_timed_events,
                self._event_keys,
                self._event_earliest_time_cache,
                self._tail,
                self._next_order,
                self._next_rank,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self._heap,
            self._timed_events,
            self._non_timed_events,
            self._event_keys,
            self._event_earliest_time_cache,
            self._tail,
            self._next_order,
            self._next_rank,
        ) = state

    def __repr__(self) -> str:
        return (
            f"Schedule(heap={self._heap}, "
            f"timed_events={self._timed_events}, "
            f"non_timed_events={self._non_timed_events}, "
            f"tail={self._tail}, "
            f"next_order={self._next_order}, "
            f"next_rank={self._next_rank}"
        )

    def reset(self) -> None:
        "Reset the schedule to its initial empty state."
        self._heap.clear()
        self._timed_events.clear()
        self._non_timed_events.clear()

        self._event_keys.clear()
        self._event_earliest_time_cache.clear()

        self._tail = None
        self._next_order = 0
        self._next_rank = 0

    def is_empty(self) -> bool:
        "Check if there are no scheduled events."
        return not self._heap

    def next_time(self) -> Time:
        "Get the next scheduled time for events."
        return self._heap[0]

    def _create_time_slot(self, time: Time) -> None:
        "Create a new time slot for events, ensuring the heap is updated."
        if time not in self._timed_events:
            self._timed_events[time] = []
            self._non_timed_events[time] = []
            heappush(self._heap, time)

    def _reschedule_non_timed_events(
        self, entries: list[_Entry], state: ScheduleState
    ) -> None:
        "Reschedule non-timed events to the next time step, preserving their order."
        if not entries:
            return

        *_, first_event = entries[0]
        time = first_event.earliest_time(state)

        if time <= state.time:
            raise ValueError(
                f"Cannot reschedule events triggered by {first_event} to the past: "
                f"{time} <= {state.time}."
            )

        self._create_time_slot(time)
        for entry in entries:
            *_, event = entry

            heappush(self._non_timed_events[time], entry)
            self._event_earliest_time_cache[event] = time

        if first_event.blocking:
            if self._tail is None or time > self._tail:
                self._tail = time

    def instruction_queue(
        self, state: ScheduleState
    ) -> Iterator[SimulationEvent]:
        """
        Get an iterator over the events that are ready to be processed, in the
        correct order according to their timing and blocking behavior.
        """
        time = heappop(self._heap)

        assert (
            time == state.time
        ), f"Next event time {time} does not match current state time {state.time}"

        timed_events = self._timed_events.pop(time, [])
        for event in timed_events:
            if not event.is_ready(state):
                raise ValueError(
                    f"Event is not ready to be processed: {event} at time {time}"
                )

            yield event

        untimed_events = self._non_timed_events.pop(time, [])
        deferred_events: list[_Entry] = []

        while untimed_events:
            entry = untimed_events[0]
            *_, event = entry

            if event.is_ready(state):
                heappop(untimed_events)
                yield event
                continue

            if event.blocking:
                self._reschedule_non_timed_events(untimed_events, state)
                break

            heappop(untimed_events)
            deferred_events.append(entry)

        for entry in deferred_events:
            self._reschedule_non_timed_events([entry], state)

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
        """
        Add an event to the schedule.

        Parameters:
        - event: The event to be scheduled.
        - state: The current schedule state, used for resolving the event and
                    calculating earliest time.
        - time: The specific time to schedule the event. If None, the event is
                scheduled as a non-timed event at the earliest possible time.
        - priority: The priority of the event, used for non-timed events. Higher
                    priority events are processed first. Only applicable for
                    non-timed events (time=None).

        Raises:
        - ValueError: If the event is already scheduled, if the time is in the
                      past, or if priority is provided for a timed event.
        """
        event.resolve(state)

        if event in self._event_keys:
            raise ValueError(f"Event is already scheduled: {event}")

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

            self._event_earliest_time_cache[event] = time
            self._event_keys[event] = (rank, -priority, order)

            self._create_time_slot(time)
            heappush(
                self._non_timed_events[time],
                (rank, -priority, order, event),
            )
            return

        if priority is not None:
            raise ValueError(
                "priority is only supported for non-timed (C) events"
            )

        if time < state.time:
            raise ValueError(
                f"Cannot schedule event in the past: {time} < {state.time}"
            )

        self._event_earliest_time_cache[event] = time

        self._create_time_slot(time)
        self._timed_events[time].append(event)

    def remove_event(self, event: SimulationEvent) -> None:
        "Remove an event from the schedule, if it is still scheduled."
        time = self._event_earliest_time_cache.pop(event)

        if event in self._event_keys:
            rank, priority, order = self._event_keys.pop(event)
            entry = (rank, priority, order, event)

            self._non_timed_events[time].remove(entry)
            heapify(self._non_timed_events[time])

        else:
            self._timed_events[time].remove(event)

    def reschedule_event(self, event: SimulationEvent, new_time: Time) -> None:
        "Reschedule an existing event to a new time."
        self.remove_event(event)

        self._create_time_slot(new_time)
        self._event_earliest_time_cache[event] = new_time
        self._timed_events[new_time].append(event)

    def change_event_priority(
        self, event: SimulationEvent, new_priority: PriorityValue
    ) -> None:
        "Change the priority of an existing non-timed event."
        if event not in self._event_keys:
            raise ValueError(f"Event is not scheduled: {event}")

        time = self._event_earliest_time_cache[event]
        rank, old_priority, order = self._event_keys[event]

        old_entry = (rank, old_priority, order, event)
        new_entry = (rank, -new_priority, order, event)

        entries = self._non_timed_events.get(time, [])
        idx = entries.index(old_entry)

        entries[idx] = new_entry
        self._event_keys[event] = (rank, -new_priority, order)
        heapify(entries)

    def clear_schedule(self) -> None:
        "Clear all scheduled events, resetting the schedule to an empty state."
        self.reset()

    def peek_events(self) -> Iterator[SimulationEvent]:
        "Peek at all scheduled events in the order they would be processed, without modifying the schedule."
        for time in sorted(self._timed_events):
            for event in self._timed_events[time]:
                yield event

        for time in sorted(self._non_timed_events):
            entries = sorted(self._non_timed_events[time])
            for *_, event in entries:
                yield event

    def peek_events_at_time(self, time: Time) -> Iterator[SimulationEvent]:
        "Peek at all scheduled events at a specific time, without modifying the schedule."
        for event in self._timed_events.get(time, []):
            yield event

        entries = sorted(self._non_timed_events.get(time, []))
        for *_, event in entries:
            yield event

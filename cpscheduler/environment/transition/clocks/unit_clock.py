"""Clock that advances by one time unit after every step."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import Time
from cpscheduler.environment.transition.actions import TimeUpdateInstruction
from cpscheduler.environment.transition.core import GlobalClock

if TYPE_CHECKING:
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.actions import Instruction
    from cpscheduler.environment.transition.core import Transition

QueueEntry = tuple[Time, float, int, "Instruction[Any]"]


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class UnitClock(GlobalClock):
    """A clock that advances by one time unit after every step.

    Only the instructions of the current time are processed, and they must be
    ready. An untimed instruction is due at the current time.
    """

    _queue: list[QueueEntry]
    _time_trail: list[Time]
    _flush_tick: bool
    _halted: bool
    _advance_target: Time

    def __init__(self) -> None:
        self._queue = []
        self._time_trail = []
        self.reset()

    @override
    def reset(self) -> None:
        super().reset()
        self._queue.clear()
        self._time_trail.clear()
        self._flush_tick = False
        self._halted = False
        self._advance_target = 0

    @override
    def push(
        self,
        instruction: Instruction[Any],
        time: Time | None,
        priority: float | None,
        event_id: int,
    ) -> None:
        if time is None:
            time = self.time

        elif time < self.time:
            raise ValueError(
                "Cannot schedule instructions for past time steps. "
                f"Current time step is {self.time}, but got {time}."
            )

        heappush(
            self._queue,
            (
                time,
                0.0 if priority is None else -priority,
                event_id,
                instruction,
            ),
        )

    @override
    def pop(
        self, state: ScheduleState, transition: Transition[Any]
    ) -> Instruction[Any] | None:
        if self._halted:
            self._halted = False
            return None

        queue = self._queue

        if queue and queue[0][0] == self.time:
            instruction = heappop(queue)[3]

            if not instruction.is_ready(state, transition):
                raise RuntimeError(
                    f"Cannot process unready instruction {instruction}."
                )

            return instruction

        # A tick that was announced ends the step after being propagated.
        if self._flush_tick:
            self._flush_tick = False
            return None

        if state.is_terminal():
            self.time = max(self.time, state.get_latest_end())
            return None

        next_time = self.time + 1
        target = self._advance_target

        # A requested advancement must not skip a timed instruction.
        if target > next_time:
            next_time = min(target, queue[0][0]) if queue else target

        if next_time >= target:
            self._advance_target = 0

        self.time = next_time

        expired = state.expire_tasks(self.time)
        if expired or transition.track_time:
            self._flush_tick = True
            return TimeUpdateInstruction(self.time)

        return None

    @override
    def is_empty(self) -> bool:
        return not self._queue

    @override
    def halt(self) -> None:
        self._halted = True

    @override
    def advance_to(self, time: Time) -> None:
        if time < self.time:
            raise RuntimeError(
                "Cannot advance to a time earlier than the current time "
                f"({time} < {self.time})."
            )

        self._advance_target = time

    @override
    def checkpoint(self, mark: int) -> None:
        self._time_trail.append(self.time)

    @override
    def backtrack(self, mark: int) -> None:
        self._queue.clear()
        self._flush_tick = False
        self._halted = False
        self._advance_target = 0

        self.time = self._time_trail[mark]
        del self._time_trail[mark:]

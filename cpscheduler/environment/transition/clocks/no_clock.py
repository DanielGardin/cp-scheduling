"""Clock that never advances, the instructions only have an order."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import Time
from cpscheduler.environment.transition.core import Clock

if TYPE_CHECKING:
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.actions import Instruction
    from cpscheduler.environment.transition.core import Transition

QueueEntry = tuple[Time, float, int, "Instruction[Any]"]


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class NoClock(Clock):
    """A clock that never advances.

    Instructions run in the order of their time, then by highest priority,
    then in insertion order. An untimed instruction has time 0. The time of an
    instruction is only an ordering key: it does not delay the task, so the
    placement alone decides where it starts.

    A blocking instruction that is not ready is an error, as nothing can make
    it ready. A non-blocking one waits in the queue for the instructions that
    make it ready, which decodes a priority list.
    """

    _queue: list[QueueEntry]
    _halted: bool

    def __init__(self) -> None:
        self._queue = []
        self.reset()

    @override
    def reset(self) -> None:
        super().reset()
        self._queue.clear()
        self._halted = False

    @override
    def push(
        self,
        instruction: Instruction[Any],
        time: Time | None,
        priority: float | None,
        event_id: int,
    ) -> None:
        heappush(
            self._queue,
            (
                0 if time is None else time,
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
        skipped: list[QueueEntry] | None = None
        found: Instruction[Any] | None = None

        while queue:
            entry = heappop(queue)
            instruction = entry[3]

            if instruction.is_ready(state, transition):
                found = instruction
                break

            # Without a clock, nothing but another instruction can make it
            # ready, and the ones behind a blocking instruction must wait.
            if instruction.blocking:
                if skipped is not None:
                    for other in skipped:
                        heappush(queue, other)

                raise RuntimeError(
                    f"Cannot process unready instruction {instruction}."
                )

            if skipped is None:
                skipped = []
            skipped.append(entry)

        if skipped is not None:
            for other in skipped:
                heappush(queue, other)

        return found

    @override
    def is_empty(self) -> bool:
        return not self._queue

    @override
    def halt(self) -> None:
        self._halted = True

    @override
    def backtrack(self, mark: int) -> None:
        self._queue.clear()
        self._halted = False

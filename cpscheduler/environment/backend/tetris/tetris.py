"""Tetris Backend class."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.backend.backend import EventID, ScheduleBackend

if TYPE_CHECKING:
    from cpscheduler.environment.backend import Instruction
    from cpscheduler.environment.constants import TaskID, Time
    from cpscheduler.environment.state import ScheduleState


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class TetrisBackend(ScheduleBackend):
    """Tetris-like dispatching kernel for managing a priority queue.

    This class is responsible for decoding instructions in the given order,
    advancing to its earliest start, instead of maintaining a global clock.
    This backend is also called Append Scheduling Generation Schema in the
    scheduling literature.
    """

    backend = "tetris"

    _queue: list[tuple[float, EventID, Instruction[TetrisBackend]]]

    @override
    def reset(self) -> None:
        super().reset()
        self._queue = []

    @override
    def is_empty(self) -> bool:
        return bool(self._queue)

    @override
    def dispatch_instruction(self, state: ScheduleState) -> Instruction | None:
        if not self._queue:
            return None

        _, _, instruction = heappop(self._queue)

        return instruction

    @override
    def get_eligible_set(self, state: ScheduleState) -> list[TaskID]:
        return state.get_unlocked_tasks()

    def add_instruction(
        self,
        instruction: Instruction[TetrisBackend],
        time: Time | None = None,
        priority: float | None = None,
    ) -> EventID:
        """Schedule a new instruction."""
        if time is not None:
            raise NotImplementedError(
                "Tetris Backend currnently does not accept timed events."
            )

        event_id = super().add_instruction(instruction, time, priority)
        priority = priority if priority is not None else 0.0

        heappush(self._queue, (-priority, event_id, instruction))

        return event_id

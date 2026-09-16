"""Time-stepped backend class."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.backend.backend import EventID, ScheduleBackend
from cpscheduler.environment.constants import TaskID, Time

if TYPE_CHECKING:
    from cpscheduler.environment.backend.actions import Instruction
    from cpscheduler.environment.state import ScheduleState


HeapQueue = list[tuple[Time, float, EventID, "Instruction[StepBackend]"]]


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class StepBackend(ScheduleBackend):
    """Backend that only accepts instructions for the current time step."""

    backend = "step"

    time: Time
    _queue: HeapQueue

    def __init__(self) -> None:
        self.reset()

    @override
    def reset(self) -> None:
        super().reset()

        self.time = 0
        self._queue = []

    @override
    def is_empty(self) -> bool:
        return not self._queue

    @override
    def get_eligible_tasks(self, state: ScheduleState) -> list[TaskID]:
        return state.get_available_tasks(self.time)

    @override
    def dispatch_instruction(self, state: ScheduleState) -> Instruction | None:
        if self._queue:
            time, _, _, instruction = self._queue[0]

            if time == self.time:
                heappop(self._queue)
                return instruction

        if state.is_terminal():
            return None

        self.time += 1
        return None

    @override
    def get_info(self) -> dict[str, int]:
        return {"current_time": self.time}

    @override
    def add_instruction(
        self,
        instruction: Instruction[StepBackend],
        time: Time | None = None,
        priority: float | None = None,
    ) -> EventID:
        """Schedule an instruction for the current time step only."""
        time = self.time if time is None else time

        if time < self.time:
            raise ValueError(
                "Cannot schedule instructions for past time steps. "
                f"Current time step is {self.time}, but got {time}."
            )

        event_id = super().add_instruction(instruction, time, priority)
        effective_priority = 0.0 if priority is None else priority
        heappush(
            self._queue,
            (time, -effective_priority, event_id, instruction),
        )

        return event_id

from typing import Self, ClassVar, Optional

from enum import Enum
from collections import deque

from .tasks import Tasks, Status


class Signal(Enum):
    """Enumeration of signals that can be sent to the scheduler"""

    Finish = 0
    Pending = 1
    Halt = 2
    Error = 3
    Advance = 4
    Skip = 5

    def is_failure(self) -> bool:
        return self in (Signal.Pending, Signal.Error)


SignalInfo   = Optional[str | int]
Instructions = ['execute', 'submit', 'pause', 'complete', 'advance', 'query', 'clear']

class Instruction:
    name: ClassVar[str]

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, deque[Self]],
    ) -> tuple[Signal, SignalInfo]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Instruction: {self.name}"


class Execute(Instruction):
    name = "execute"

    def __init__(self, task_id: int, machine: int = 0):
        self.task_id = task_id
        self.machine = machine

    def __repr__(self) -> str:
        return super().__repr__() + f" task {self.task_id} on machine {self.machine}"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, deque[Self]],
    ) -> tuple[Signal, SignalInfo]:
        task = tasks[self.task_id]
        if task.is_available(current_time):
            task.execute(current_time, self.machine)

            return Signal.Finish, None

        status = task.get_status(current_time)
        if status == Status.EXECUTING or status == Status.PAUSED:
            return (
                Signal.Error,
                f"Task {self.task_id} cannot be executed. It is already being executed or completed",
            )

        for fixed_task in tasks:
            if fixed_task.is_executing(current_time):
                return Signal.Pending, None

        return Signal.Halt, None


class Submit(Instruction):
    name = "submit"

    def __init__(self, task_id: int, machine: int = 0):
        self.task_id = task_id
        self.machine = machine

    def __repr__(self) -> str:
        return super().__repr__() + f" task {self.task_id} on machine {self.machine}"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, deque[Self]],
    ) -> tuple[Signal, SignalInfo]:
        task = tasks[self.task_id]
        if task.is_available(current_time):
            task.execute(current_time, self.machine)

            return Signal.Finish, None

        status = task.get_status(current_time)
        if status == Status.EXECUTING or status == Status.PAUSED:
            return (
                Signal.Error,
                f"Task {self.task_id} cannot be executed. It is already being executed or completed",
            )

        return Signal.Skip, None


class Pause(Instruction):
    name = "pause"

    def __init__(self, task_id: int):
        self.task_id = task_id

    def __repr__(self) -> str:
        return super().__repr__() + f" task {self.task_id}"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, deque[Self]],
    ) -> tuple[Signal, SignalInfo]:
        task = tasks[self.task_id]
        status = task.get_status(current_time)

        if status == Status.EXECUTING:
            task.pause(current_time)

            return Signal.Finish, None

        if status == Status.COMPLETED:
            return Signal.Error, f"Task {self.task_id} already terminated"

        return Signal.Pending, None


class Complete(Instruction):
    name = "complete"

    def __init__(self, task_id: int):
        self.task_id = task_id

    def __repr__(self) -> str:
        return super().__repr__() + f" task {self.task_id}"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, deque[Self]],
    ) -> tuple[Signal, SignalInfo]:
        task = tasks[self.task_id]
        if task.is_executing(current_time):
            return Signal.Advance, task.get_end()

        if task.is_completed(current_time):
            return Signal.Error, f"Task {self.task_id} already terminated"

        return Signal.Pending, None


class Advance(Instruction):
    name = "advance"

    def __init__(self, time: int):
        self.time = time

    def __repr__(self) -> str:
        return super().__repr__() + f" by {self.time} units"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, deque[Self]],
    ) -> tuple[Signal, SignalInfo]:
        return Signal.Advance, current_time + self.time


class Query(Instruction):
    name = "query"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, deque[Self]],
    ) -> tuple[Signal, SignalInfo]:
        return Signal.Halt, None


class Clear(Instruction):
    name = "clear"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, deque[Self]],
    ) -> tuple[Signal, SignalInfo]:
        scheduled_instructions.clear()
        scheduled_instructions[-1] = deque()

        return Signal.Finish, None

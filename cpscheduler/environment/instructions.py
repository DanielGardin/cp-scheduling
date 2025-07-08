"""
instructions.py

This module defines the instructions that can be executed in the scheduling environment.
Instructions are used to control the execution of tasks, manage their states, and interact
with the scheduler.
"""

from typing import ClassVar, Final, TypeAlias, Iterable
from typing_extensions import Unpack, TypeIs

from dataclasses import dataclass

from mypy_extensions import mypyc_attr, u8

from ._common import TASK_ID, TIME, MACHINE_ID, Int
from .tasks import Tasks

SingleAction: TypeAlias = tuple["str | Instruction", Unpack[tuple[Int, ...]]]
ActionType: TypeAlias = SingleAction | Iterable[SingleAction] | None

def is_single_action(
    action: ActionType,
) -> TypeIs[SingleAction]:
    "Check if the action is a single instruction or a iterable of instructions."
    if not isinstance(action, tuple):
        return False

    return isinstance(action[0], str) and all(isinstance(arg, Int) for arg in action[1:])

# Flags are not supported by mypyc yet
class Action:
    "Flags for possible actions made by the scheduler in response to an instruction."

    # Tell the scheduler the instruction was skiped, the default behavior is to remove
    # the instruction after processing it, but this flag disables that
    SKIPPED: Final[u8] = 1

    # The scheduler should reevaluate the current bounds (used when preempting tasks)
    REEVALUATE: Final[u8] = 2

    # The scheduler should propagate the constraints further
    PROPAGATE: Final[u8] = 4

    # The scheduler should advance the time
    ADVANCE: Final[u8] = 8

    # The scheduler should advance to the next decision point
    ADVANCE_NEXT: Final[u8] = 16

    # The scheduler should raise an exception
    RAISE: Final[u8] = 32

    # The scheduler should stop processing instructions
    HALT: Final[u8] = 64

    DONE: Final[u8] = PROPAGATE | ADVANCE
    ERROR: Final[u8] = RAISE | HALT
    WAIT: Final[u8] = SKIPPED | ADVANCE_NEXT


@dataclass
class Signal:
    "Action signal with additional parameters."

    action: u8
    time: TIME = 0
    info: str = ""


@mypyc_attr(allow_interpreted_subclasses=True)
class Instruction:
    """
    Base class for all instructions in the scheduling environment.

    Instructions are used to control the execution of tasks, manage their states, and interact
    with the scheduler. Each instruction has a name and a method to process it.

    You can create custom instructions by subclassing this class and implementing the
    `process` method.
    Caution: Instructions directly manipulate the state of tasks and the scheduler, so
    new ones should be implemented carefully.
    """

    name: ClassVar[str]

    def process(
        self,
        current_time: TIME,
        tasks: Tasks,
        scheduled_instructions: dict[TIME, list["Instruction"]],
    ) -> Signal:
        "Process the instruction at the given current time."
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Instruction: {self.name}"


class Execute(Instruction):
    "Executes a task on a specific machine. If the task cannot be executed, it is waited for."

    name = "execute"

    def __init__(self, task_id: TASK_ID, machine: MACHINE_ID = 0):
        self.task_id = task_id
        self.machine = machine

    def __repr__(self) -> str:
        return super().__repr__() + f" task {self.task_id} on machine {self.machine}"

    def process(
        self,
        current_time: TIME,
        tasks: Tasks,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        task = tasks[self.task_id]
        if task.is_available(current_time, self.machine):
            tasks.fix_task(self.task_id, self.machine, current_time)

            return Signal(Action.DONE)

        if task.is_fixed():
            return Signal(
                Action.RAISE,
                info=f"Task {self.task_id} cannot be executed. It is already being executed or completed",
            )

        return Signal(Action.WAIT)


class Submit(Instruction):
    "Executes a task on a specific machine. If the task cannot be executed, skips it."

    name = "submit"

    def __init__(self, task_id: TASK_ID, machine: MACHINE_ID = 0):
        self.task_id = task_id
        self.machine = machine

    def __repr__(self) -> str:
        return super().__repr__() + f" task {self.task_id} on machine {self.machine}"

    def process(
        self,
        current_time: TIME,
        tasks: Tasks,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        task = tasks[self.task_id]
        if task.is_available(current_time, self.machine):
            tasks.fix_task(self.task_id, self.machine, current_time)

            return Signal(Action.DONE)

        if task.is_fixed():
            return Signal(
                Action.ERROR,
                info=f"Task {self.task_id} cannot be executed. It is already being executed or completed",
            )

        return Signal(Action.SKIPPED)


class Pause(Instruction):
    "Pauses a task if it is currently executing. Can only be used in preemptive scheduling."

    name = "pause"

    def __init__(self, task_id: TASK_ID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return super().__repr__() + f" task {self.task_id}"

    def process(
        self,
        current_time: TIME,
        tasks: Tasks,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        task = tasks[self.task_id]

        if task.is_executing(current_time):
            tasks.unfix_task(self.task_id, current_time)

            return Signal(Action.DONE | Action.REEVALUATE)

        if task.is_completed(current_time):
            return Signal(Action.ERROR, info=f"Task {self.task_id} already terminated")

        return Signal(Action.WAIT)


class Complete(Instruction):
    "Advances the current time to the end of an executing task."

    name = "complete"

    def __init__(self, task_id: TASK_ID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return super().__repr__() + f" task {self.task_id}"

    def process(
        self,
        current_time: TIME,
        tasks: Tasks,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        task = tasks[self.task_id]
        if task.is_executing(current_time):
            return Signal(Action.ADVANCE, task.get_end())

        if task.is_completed(current_time):
            return Signal(Action.ERROR, info=f"Task {self.task_id} already terminated")

        return Signal(Action.WAIT)


class Advance(Instruction):
    "Advances the current time by a specified amount or to the next decision point if not specified."

    name = "advance"

    def __init__(self, time: TIME = -1):
        self.time = time

    def __repr__(self) -> str:
        if self.time == -1:
            return super().__repr__() + " to the next decision point"

        return super().__repr__() + f" by {self.time} units"

    def process(
        self,
        current_time: TIME,
        tasks: Tasks,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        if self.time == -1:
            return Signal(Action.ADVANCE_NEXT)

        return Signal(Action.ADVANCE, current_time + self.time)


class Query(Instruction):
    "When processed, halts the environment and returns its current state."

    name = "query"

    def process(
        self,
        current_time: TIME,
        tasks: Tasks,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        return Signal(Action.HALT)


class Clear(Instruction):
    "Clears all upcoming instructions and resets the schedule."

    name = "clear"

    def process(
        self,
        current_time: TIME,
        tasks: Tasks,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        scheduled_instructions.clear()
        scheduled_instructions[-1] = list()

        return Signal(Action.DONE)


def parse_args(
    instruction_name: str,
    args: tuple[int, ...],
    n_required: int,
) -> tuple[tuple[int, ...], int]:
    "Parse the raw instruction arguments into required and optional arguments."
    if len(args) == n_required:
        return args, -1

    if len(args) == n_required + 1:
        return args[:-1], args[-1]

    raise ValueError(
        f"Expected {n_required} or {n_required + 1} arguments for instruction {instruction_name},"
        "got {len(args)}."
    )


def parse_instruction(
    action: str | Instruction, args: tuple[int, ...]
) -> tuple[Instruction, TIME]:
    "Parse raw instruction arguments into an Instruction object and the scheduled time."
    instruction: Instruction

    if isinstance(action, Instruction):
        _, time = parse_args(action.name, args, 0)
        instruction = action

    elif action == "execute":
        (task_id, machine), time = parse_args(action, args, 2)

        instruction = Execute(task_id, machine)

    elif action == "submit":
        (task_id, machine), time = parse_args(action, args, 2)

        instruction = Submit(task_id, machine)

    elif action == "pause":
        (task_id,), time = parse_args(action, args, 1)

        instruction = Pause(task_id)

    elif action == "complete":
        (task_id,), time = parse_args(action, args, 1)

        instruction = Complete(task_id)

    elif action == "advance":
        (to_time,), time = parse_args(action, args, 1)

        instruction = Advance(to_time)

    elif action == "query":
        _, time = parse_args(action, args, 0)
        instruction = Query()

    elif action == "clear":
        _, time = parse_args(action, args, 0)
        instruction = Clear()

    else:
        raise ValueError(f"Unknown instruction {action}")

    return instruction, time

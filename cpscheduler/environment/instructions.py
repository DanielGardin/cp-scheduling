from typing import Self, ClassVar, Final

from dataclasses import dataclass

from .tasks import Tasks, Status

# Flags are not supported by mypyc
class Action:
    SKIPPED: Final[int]      = 1 # Tell the scheduler the instruction was skiped
    REEVALUATE: Final[int]   = 2 # The scheduler should reevaluate the current bounds
    PROPAGATE: Final[int]    = 4 # The scheduler should propagate the constraints further
    ADVANCE: Final[int]      = 8 # The scheduler should advance the time
    ADVANCE_NEXT: Final[int] = 16 # The scheduler should advance to the next decision point
    RAISE: Final[int]        = 32 # The scheduler should raise an exception
    HALT: Final[int]         = 64 # The scheduler should stop processing instructions

    DONE: Final[int]  = PROPAGATE | ADVANCE
    ERROR :Final[int] = RAISE | HALT
    WAIT: Final[int]  = SKIPPED | PROPAGATE | ADVANCE_NEXT

@dataclass
class Signal:
    action: int
    param: int = 0
    info: str = ""

class Instruction:
    name: ClassVar[str]

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, list[Self]],
    ) -> Signal:
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
        scheduled_instructions: dict[int, list[Self]],
    ) -> Signal:
        task = tasks[self.task_id]
        if task.is_available(current_time):
            task.execute(current_time, self.machine)

            return Signal(Action.DONE)

        status = task.get_status(current_time)
        if status == Status.EXECUTING or status == Status.PAUSED:
            return Signal(
                Action.RAISE,
                info=f"Task {self.task_id} cannot be executed. It is already being executed or completed"
            )

        for fixed_task in tasks:
            if fixed_task.is_executing(current_time):
                return Signal(Action.WAIT)

        return Signal(Action.HALT)


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
        scheduled_instructions: dict[int, list[Self]],
    ) -> Signal:
        task = tasks[self.task_id]
        if task.is_available(current_time):
            task.execute(current_time, self.machine)

            return Signal(Action.DONE)

        status = task.get_status(current_time)
        if status == Status.EXECUTING or status == Status.PAUSED:
            return Signal(
                Action.ERROR,
                info=f"Task {self.task_id} cannot be executed. It is already being executed or completed",
            )

        return Signal(Action.SKIPPED)


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
        scheduled_instructions: dict[int, list[Self]],
    ) -> Signal:
        task = tasks[self.task_id]
        status = task.get_status(current_time)

        if status == Status.EXECUTING:
            task.pause(current_time)

            return Signal(Action.DONE | Action.REEVALUATE)

        if status == Status.COMPLETED:
            return Signal(Action.ERROR, info=f"Task {self.task_id} already terminated")

        return Signal(Action.WAIT)


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
        scheduled_instructions: dict[int, list[Self]],
    ) -> Signal:
        task = tasks[self.task_id]
        if task.is_executing(current_time):
            return Signal(Action.ADVANCE, task.get_end())

        if task.is_completed(current_time):
            return Signal(Action.ERROR, info=f"Task {self.task_id} already terminated")

        return Signal(Action.WAIT)


class Advance(Instruction):
    name = "advance"

    def __init__(self, time: int):
        self.time = time

    def __repr__(self) -> str:
        if self.time == -1:
            return super().__repr__() + " to the next decision point"

        return super().__repr__() + f" by {self.time} units"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, list[Self]],
    ) -> Signal:
        if self.time == -1:
            return Signal(Action.ADVANCE_NEXT)

        return Signal(Action.ADVANCE, current_time + self.time)


class Query(Instruction):
    name = "query"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, list[Self]],
    ) -> Signal:
        return Signal(Action.HALT)


class Clear(Instruction):
    name = "clear"

    def process(
        self,
        current_time: int,
        tasks: Tasks,
        scheduled_instructions: dict[int, list[Self]],
    ) -> Signal:
        scheduled_instructions.clear()
        scheduled_instructions[-1] = list()

        return Signal(Action.DONE)



def parse_args(
    instruction_name: str,
    args: tuple[int, ...],
    n_required: int,
) -> tuple[tuple[int, ...], int]:
    if len(args) == n_required:
        return args, -1

    elif len(args) == n_required + 1:
        return args[:-1], args[-1]

    else:
        raise ValueError(
            f"Expected {n_required} or {n_required + 1} arguments for instruction {instruction_name} , got {len(args)}."
        )

def parse_instruction(action: str | Instruction, args: tuple[int, ...]) -> tuple[Instruction, int]:
    instruction: Instruction

    match action:
        case "execute":
            (task_id, machine), time = parse_args(action, args, 2)

            instruction = Execute(task_id, machine)
        
        case "submit":
            (task_id, machine), time = parse_args(action, args, 2)

            instruction = Submit(task_id, machine)
        
        case "pause":
            (task_id,), time = parse_args(action, args, 1)

            instruction = Pause(task_id)
        
        case "complete":
            (task_id,), time = parse_args(action, args, 1)

            instruction = Complete(task_id)
        
        case "advance":
            (to_time,), time = parse_args(action, args, 1)

            instruction = Advance(to_time)
        
        case "query":
            _, time = parse_args(action, args, 0)
            instruction = Query()
        
        case "clear":
            _, time = parse_args(action, args, 0)
            instruction = Clear()
        
        case Instruction():
            _, time = parse_args(action.name, args, 0)
            instruction = action

        case _:
            raise ValueError(f"Unknown instruction {action}")

    return instruction, time
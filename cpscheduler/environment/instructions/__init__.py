__all__ = [
    "Schedule",
    "InstructionResult",
    "QueueControl",
    "Instruction",
    "SingleAction",
    "ActionType",
    "is_single_action",
    # Instruction implementations
    "Execute",
    "Submit",
    "ExecuteJob",
    "SubmitJob",
    "Pause",
    "Resume",
    "Complete",
    "Advance",
    "Noop",
    "Query",
    "Clear",
]

from .instructions import *
from .schedule import (
    Schedule,
    InstructionResult,
    QueueControl,
    Instruction,
    DEFAULT_QUEUE_TIME,
)

from typing import TypeAlias
from typing_extensions import Unpack, TypeIs
from collections.abc import Iterable
from cpscheduler.environment.constants import Int, Time

SingleAction: TypeAlias = tuple["str | Instruction", Unpack[tuple[Int, ...]]]
ActionType: TypeAlias = SingleAction | Iterable[SingleAction] | None


def is_single_action(
    action: ActionType,
) -> TypeIs[SingleAction]:
    "Check if the action is a single instruction or a iterable of instructions."
    if not isinstance(action, tuple):
        return False

    return isinstance(action[0], (str, Instruction))


from cpscheduler.environment.state import ScheduleState
from cpscheduler.utils.list_utils import convert_to_list


def parse_args(
    args: list[Int],
    output_size: int,
    instruction_name: str = "Instruction",
) -> list[int]:
    "Parse the raw instruction arguments into required and optional arguments."
    if len(args) > output_size:
        raise ValueError(
            f"Too many arguments provided for instruction {instruction_name}. "
            f"Expected at most {output_size}, got {len(args)}."
        )

    return convert_to_list(args, int) + [-1] * (output_size - len(args))


def parse_instruction(
    instruction: str | Instruction, args: list[Int], state: ScheduleState
) -> tuple[Instruction, Time]:
    "Parse raw instruction arguments into an Instruction object and the scheduled time."

    return_instruction: Instruction
    match instruction:
        case "execute":
            if 0 < len(args) <= 2:
                task_id, time_or_machine = parse_args(args, 2, "execute")

                machines = state.instance.get_machines(task_id)

                if len(machines) == 1:
                    # Consider the second argument as time
                    machine = next(iter(machines))
                    time = time_or_machine

                else:
                    # Consider the second argument as machine
                    machine = time_or_machine
                    time = -1

            else:
                task_id, machine, time = parse_args(args, 3, "execute")

            return_instruction = Execute(task_id, machine)

        case "submit":
            if 0 < len(args) <= 2:
                task_id, time_or_machine = parse_args(args, 2, "submit")

                machines = state.instance.get_machines(task_id)

                if len(machines) == 1:
                    # Consider the second argument as time
                    machine = next(iter(machines))
                    time = time_or_machine

                else:
                    # Consider the second argument as machine
                    machine = time_or_machine
                    time = -1

            else:
                task_id, machine, time = parse_args(args, 3, "submit")

            return_instruction = Submit(task_id, machine)

        case "execute job":
            job_id, machine, time = parse_args(args, 3, "execute job")

            return_instruction = ExecuteJob(job_id, machine)

        case "submit job":
            job_id, machine, time = parse_args(args, 3, "submit job")

            return_instruction = SubmitJob(job_id, machine)

        case "pause":
            task_id, time = parse_args(args, 2, "pause")

            return_instruction = Pause(task_id)

        case "resume":
            task_id, time = parse_args(args, 2, "resume")

            return_instruction = Resume(task_id)

        case "complete":
            task_id, time = parse_args(args, 2, "complete")

            return_instruction = Complete(task_id)

        case "advance":
            to_time, time = parse_args(args, 2, "advance")

            return_instruction = Advance(to_time)

        case "noop":
            (time,) = parse_args(args, 1, "noop")

            return_instruction = Noop()

        case "query":
            (time,) = parse_args(args, 1, "query")

            return_instruction = Query()

        case "clear":
            (time,) = parse_args(args, 1, "clear")

            return_instruction = Clear()

        case Instruction():
            (time,) = parse_args(args, 1, "Instruction")

            return_instruction = instruction

        case _:
            raise ValueError(f"Unknown instruction {instruction}")

    if time < state.time and time != DEFAULT_QUEUE_TIME:
        raise ValueError(
            f"Cannot schedule instruction {return_instruction} in the past. "
            f"Current time is {state.time}, got {time}."
        )

    return return_instruction, time

from typing import Any, TypeAlias
from typing_extensions import Unpack, TypeIs, TypedDict, NotRequired
from collections.abc import Iterable

from cpscheduler.environment.constants import Int, Time

from cpscheduler.environment.des.base import (
    SimulationEvent,
    PriorityValue,
    instructions
)

Instruction = SimulationEvent

class InstructionKwargs(TypedDict):
    time: NotRequired[Int]
    priority: NotRequired[Int]

InstructionArgs = Int | InstructionKwargs

BAction: TypeAlias = (
    tuple[InstructionArgs, Instruction]
    | tuple[InstructionArgs, str, Unpack[tuple[Int, ...]]]
    # Mypy does not support Any in Unpack, so we use Int as a placeholder
)
"Timed instruction action, represented as a tuple of (time, instruction) or (time, instruction_name, *args)."

CAction: TypeAlias = Instruction | tuple[str, Unpack[tuple[Int, ...]]]
"Instruction action, represented as an Instruction object or a tuple of (instruction_name, *args)."

SingleAction: TypeAlias = BAction | CAction

ActionType: TypeAlias = SingleAction | Iterable[SingleAction] | None


def is_single_action(
    action: ActionType,
) -> TypeIs[SingleAction]:
    "Check if the action is a single instruction or a iterable of instructions."
    if not isinstance(action, tuple):
        return False

    if isinstance(action[0], int) or isinstance(action[0], dict):
        return isinstance(action[1], (str, Instruction))

    return isinstance(action[0], (str, Instruction))


def _parse_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    "Parse raw instruction arguments, converting Int to int where appropriate."
    return tuple(int(arg) if isinstance(arg, Int) else arg for arg in args)


def parse_instruction(
    instruction_args: SingleAction,
) -> tuple[Instruction, Time | None, PriorityValue | None]:
    """Parse raw instruction arguments into (Instruction, time, priority).
    
    Priority defaults to None, allowing Schedule.add_event to apply its default (0).
    """
    if isinstance(instruction_args, SimulationEvent):
        return instruction_args, None, None

    if isinstance(instruction_args[0], str):
        instruction_cls = instructions[instruction_args[0]]
        args = _parse_args(instruction_args[1:])
        return instruction_cls(*args), None, None

    # B-event or dict-prefixed: extract time and priority
    if isinstance(instruction_args[0], dict):
        kwargs: InstructionKwargs = instruction_args[0]
        time = Time(kwargs["time"]) if "time" in kwargs else None
        priority = PriorityValue(kwargs["priority"]) if "priority" in kwargs else None

    else:
        time = Time(instruction_args[0])
        priority = None

    instruction_dispatch = instruction_args[1]
    args = _parse_args(instruction_args[2:])

    if isinstance(instruction_dispatch, str):
        instruction = instructions[instruction_dispatch](*args)

    elif isinstance(instruction_dispatch, SimulationEvent):
        instruction = instruction_dispatch

    else:
        raise ValueError(
            f"Invalid instruction dispatch: {instruction_dispatch}. "
            f"Must be either an instruction name or an Instruction object."
        )

    return instruction, time, priority

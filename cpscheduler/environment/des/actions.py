from typing import Any, cast
from typing_extensions import Unpack, TypeIs, TypedDict, NotRequired
from collections.abc import Iterable

from cpscheduler.environment.constants import Int, Time

from cpscheduler.environment.des.base import (
    SimulationEvent,
    PriorityValue,
    instructions,
)


class InstructionKwargs(TypedDict):
    time: NotRequired[Int]
    priority: NotRequired[Int]


SchedulerArgs = Int | InstructionKwargs
InstructionSpec = str | type[SimulationEvent]
InstructionArgs = tuple[Int, ...]
# Mypy does not support Any in Unpack, so we use Int as a placeholder

BAction = tuple[SchedulerArgs, InstructionSpec, Unpack[InstructionArgs]]
"Timed instruction action, represented as a tuple of (time, instruction_name, *args)."

CAction = tuple[InstructionSpec, Unpack[InstructionArgs]]
"Instruction action, represented as an Instruction object or a tuple of (instruction_name, *args)."

SingleInstruction = BAction | CAction

ActionType = SingleInstruction | Iterable[SingleInstruction] | None


def is_single_action(
    action: Any,
) -> TypeIs[SingleInstruction]:
    "Check if the action is a single instruction or a iterable of instructions."
    if not isinstance(action, tuple):
        return False

    if isinstance(action[0], int) or isinstance(action[0], dict):
        spec = action[1]

    else:
        spec = action[0]

    if isinstance(spec, str):
        return True

    return isinstance(spec, type) and issubclass(spec, SimulationEvent)

def _parse_args(args: list[Any]) -> tuple[Any, ...]:
    "Parse raw instruction arguments, converting Int to int where appropriate."
    return tuple(int(arg) if isinstance(arg, Int) else arg for arg in args)


def parse_instruction(
    instruction: SingleInstruction
) -> tuple[SimulationEvent, Time | None, PriorityValue | None]:
    time: Time | None = None
    priority: PriorityValue | None = None

    if isinstance(instruction[0], (int, dict)):
        # instruction = cast(BAction, instruction)

        s_args, spec, *spec_args = instruction

        if isinstance(s_args, Int):
            time = Time(s_args)
 
        else:
            time = (
                Time(s_args["time"])
                if "time" in s_args
                else None
            )
            priority = (
                PriorityValue(s_args["priority"])
                if "priority" in s_args
                else None
            )

    else:
        instruction = cast(CAction, instruction)

        spec, *spec_args = instruction

    args = _parse_args(spec_args)

    if isinstance(spec, str):
        cls = instructions[spec]

        return cls(*args), time, priority

    return spec(*args), time, priority

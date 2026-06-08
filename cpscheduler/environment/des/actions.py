"""Common types and utilities for DES actions and instructions."""

from collections.abc import Iterable
from typing import Any, cast

from typing_extensions import TypeIs, Unpack

from cpscheduler.environment.constants import Int, Time
from cpscheduler.environment.des.base import (
    PriorityValue,
    SimulationEvent,
    instructions,
)

SchedulerArgs = Int
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
    """Check if the action is a single instruction or a iterable of instructions."""
    if not isinstance(action, tuple):
        return False

    spec = action[1] if isinstance(action[0], Int) else action[0]

    if isinstance(spec, str):
        return True

    return isinstance(spec, type) and issubclass(spec, SimulationEvent)


def _parse_args(args: list[Any]) -> tuple[Any, ...]:
    """Parse raw instruction arguments, converting Int to int where appropriate."""
    return tuple(int(arg) if isinstance(arg, Int) else arg for arg in args)


def parse_instruction(
    instruction: SingleInstruction,
) -> tuple[SimulationEvent, Time | None, PriorityValue | None]:
    """Parse a single instruction action into a SimulationEvent.

    Parameters
    ----------
    instruction : SingleInstruction
        The instruction to parse, which can be either a BAction or a CAction,
        following the formats defined below:
        - BAction: (time, instruction_name, *args)
        - CAction: (instruction_name, *args)

    Returns
    -------
    event : SimulationEvent
        The parsed SimulationEvent object corresponding to the instruction.

    time : Time | None
        The time at which the event should occur, if specified in a BAction.
        None for CAction.

    priority : PriorityValue | None
        The priority of the event, if specified in the instruction arguments.
        None if not specified.


    Notes
    -----
    Priority is currently not supported in the instruction format.
    Future versions may include priority as an optional argument in the instruction.

    """
    time: Time | None = None

    if isinstance(instruction[0], Int):
        instruction = cast("BAction", instruction)

        s_args, spec, *spec_args = instruction
        time = Time(s_args)

    else:
        instruction = cast("CAction", instruction)

        spec, *spec_args = instruction

    args = _parse_args(spec_args)

    if isinstance(spec, str):
        cls = instructions[spec]

        return cls(*args), time, None

    return spec(*args), time, None

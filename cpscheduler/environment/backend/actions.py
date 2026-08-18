"""Common interface for actions in all schedule backends."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, cast

from mypy_extensions import mypyc_attr
from typing_extensions import Self, TypeIs, TypeVar, Unpack

from cpscheduler.environment.constants import EzPickle, Int, Time

if TYPE_CHECKING:
    from cpscheduler.environment.backend.backend import ScheduleBackend
    from cpscheduler.environment.state import ScheduleState

instructions: dict[str, dict[str, type[Instruction[Any]]]] = {}

B_contra = TypeVar(
    "B_contra", bound="ScheduleBackend", default=Any, contravariant=True
)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Instruction(EzPickle, Generic[B_contra]):
    """Intermediate representation of an action in the environment.

    To create a new event, subclass this class to define the instruction type
    and behavior in the simulation.

    Attributes
    ----------
    spec: str
        Name of instruction used when parsing instruction tuples.

    backend: str
        Name of the backend that supports this instruction.

    """

    backend: ClassVar[str] = ""

    def resolve(self, state: ScheduleState) -> Self:
        """Return the resolved instruction. Defaults to self."""
        return self

    def process(self, state: ScheduleState, backend: B_contra) -> None:
        """Process the instruction, modifying the schedule state accordingly."""


def register_instruction(
    instruction: type[Instruction[Any]], spec: str, backend: str
) -> None:
    """Register an instruction as an action for a backend."""
    instruction.backend = backend
    instructions.setdefault(backend, {})[spec] = instruction


def validate_instruction(
    instruction: Instruction, state: ScheduleState, max_depth: int = 100
) -> Instruction:
    """Resolve the instruction to a fixed point."""
    for _ in range(max_depth):
        validated_instruction = instruction.resolve(state)

        if instruction is validated_instruction:
            return instruction

        instruction = validated_instruction

    raise RuntimeError(
        f"Instruction {instruction} failed to reach a fixed point after "
        f"{max_depth} resolutions."
    )


SchedulerArgs = Int
InstructionSpec = str | type[Instruction[Any]]
InstructionArgs = tuple[Int, ...]
# Mypy does not support Any in Unpack, so we use Int as a placeholder

BAction = tuple[SchedulerArgs, InstructionSpec, Unpack[InstructionArgs]]
"Timed instruction action, represented as a tuple of (time, instruction_name, *args)."

CAction = tuple[InstructionSpec, Unpack[InstructionArgs]]
"Instruction action, represented as an Instruction object or a tuple of (instruction_name, *args)."

SingleAction = BAction | CAction
ActionType = SingleAction | Iterable[SingleAction] | None


def is_single_action(
    action: Any,
) -> TypeIs[SingleAction]:
    """Check if the action is a single instruction or a iterable of instructions."""
    if not isinstance(action, tuple):
        return False

    spec = action[1] if isinstance(action[0], Int) else action[0]

    if isinstance(spec, str):
        return True

    return isinstance(spec, type) and issubclass(spec, Instruction)


def _parse_args(args: list[Any]) -> tuple[Any, ...]:
    """Parse raw instruction arguments, converting Int to int where appropriate."""
    return tuple(int(arg) if isinstance(arg, Int) else arg for arg in args)


def parse_instruction(
    action: SingleAction, backend: str
) -> tuple[Instruction[Any], Time | None, float | None]:
    """Parse a single action action into an Instruction.

    Parameters
    ----------
    action : SingleAction
        The action to parse, which can be either a BAction or a CAction,
        following the formats defined below:
        - BAction: (time, instruction_name, *args)
        - CAction: (instruction_name, *args)

    backend: str
        The backend used for dispatching in the current environment.

    Returns
    -------
    instruction : Instruction
        The parsed Instruction object corresponding to the action.

    time : Time, optional
        The time at which the instruction should be processed.
        If specified, the instruction is a BAction, otherwise, a CAction.

    priority : float | None
        The priority of the event, if specified in the instruction arguments.
        None if not specified.


    Notes
    -----
    Priority is currently not supported in the instruction format.
    Future versions may include priority as an optional argument in the instruction.

    """
    time: Time | None = None

    if isinstance(action[0], Int):
        s_args, spec, *spec_args = cast("BAction", action)
        time = Time(s_args)

    else:
        spec, *spec_args = cast("CAction", action)

    args = _parse_args(spec_args)

    if isinstance(spec, str):
        instruction_set = instructions[backend]

        if spec not in instruction_set:
            raise ValueError(
                f"Instruction '{spec}' is not defined for backend {backend}."
            )

        cls = instruction_set[spec]

        return cls(*args), time, None

    if spec.backend != backend:
        text_args = ", ".join(str(arg) for arg in args)
        raise RuntimeError(
            f"Instruction {spec.__name__}({text_args}) is only defined for "
            f"backend {spec.backend}, which is incompatible with {backend}."
        )

    return spec(*args), time, None

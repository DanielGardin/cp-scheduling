"""Instructions, the intermediate representation of the actions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, cast

from mypy_extensions import mypyc_attr
from typing_extensions import Self, TypeIs, TypeVar, Unpack

from cpscheduler.environment.constants import Int, Time
from cpscheduler.environment.mixins import EzPickle
from cpscheduler.environment.transition.core import Clock

if TYPE_CHECKING:
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.core import Transition

instructions: dict[str, type[Instruction[Any]]] = {}

C = TypeVar("C", bound="Clock")
C_contra = TypeVar("C_contra", bound="Clock", default=Any, contravariant=True)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Instruction(EzPickle, Generic[C_contra]):
    """Intermediate representation of an action in the environment.

    To create a new instruction, subclass this class and define how it changes
    the state in `process`. The type parameter is the clock that `process`
    sees in the transition: `Instruction[Clock]` for any clock, or a more
    specific one such as `Instruction[GlobalClock]`.

    An instruction that needs a specific clock must be registered with it, see
    `register_instruction`. One that is not registered is accepted by every
    transition.
    """

    # By default, instructions work on all clocks.
    clock: ClassVar[type[Clock]] = Clock

    blocking: ClassVar[bool] = False
    """Whether the instructions queued after this one wait for it.

    A blocking instruction that is not ready holds the queue until the clock
    makes it ready. A non-blocking one is skipped, and the next ready
    instruction of the queue is processed instead.
    """

    def resolve(self, state: ScheduleState) -> Self:
        """Return the resolved instruction. Defaults to self.

        This step is meant to resolve the instruction parameters statically, at
        push time.
        Raising errors during this step prevents the transition to parse incorrect
        instructions.
        """
        return self

    def is_ready(
        self, state: ScheduleState, transition: Transition[C_contra]
    ) -> bool:
        """Return whether the instruction can be processed in this state."""
        return True

    def earliest_time(self, state: ScheduleState) -> Time | None:
        """Return the earliest time at which the instruction can be ready.

        Clocks that advance use it to wait for an instruction that is not
        ready. None means that time alone does not make it ready.
        """
        return None

    def is_locked(
        self, state: ScheduleState, transition: Transition[C_contra]
    ) -> bool:
        """Return whether only another instruction can make this one ready.

        A locked instruction is a non-ready instruction that cannot start
        before another instruction unlocks it.
        When an instruction is locked, its earliest time is a lower bound for
        when it can be ready, but it is not guaranteed to be ready at that time.
        Defaults to whether `earliest_time` is None.
        """
        return self.earliest_time(state) is None

    def process(
        self, state: ScheduleState, transition: Transition[C_contra]
    ) -> None:
        """Process the instruction, changing the state accordingly."""

    def semantic(
        self, state: ScheduleState, transition: Transition[C_contra]
    ) -> dict[str, Any] | None:
        """Return the semantic metadata of this instruction, if any."""
        return None


class TimeUpdateInstruction(Instruction[Clock]):
    """Internal instruction that propagates a clock advancement to the state."""

    time: Time

    def __init__(self, time: Time) -> None:
        self.time = time

    def process(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> None:
        """Announce the new time to the time-aware components."""
        state.tight_global_time(self.time)


def register_instruction(
    instruction: type[Instruction[C]], spec: str, clock: type[C]
) -> None:
    """Register an instruction as an action for the transitions of a clock.

    Parameters
    ----------
    instruction: type[Instruction]
        The instruction class.

    spec: str
        The name of the instruction in the action tuples.

    clock: type[Clock]
        The clock class that the instruction needs. A transition accepts the
        instruction when its clock is an instance of it, so `Clock` makes it
        available everywhere.

    """
    original_clock = instruction.clock
    if not issubclass(clock, original_clock):
        raise ValueError(
            f"Instruction {instruction.__name__} was defined for "
            f"{original_clock.__name__}, but registered for {clock.__name__}, "
            f"which is not a subtype of {original_clock.__name__}."
        )

    instruction.clock = clock
    instructions[spec] = instruction


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
    if not isinstance(action, tuple) or not action:
        return False

    spec = action[1] if isinstance(action[0], Int) else action[0]

    if isinstance(spec, str):
        return True

    return isinstance(spec, type) and issubclass(spec, Instruction)


def _parse_args(args: list[Any]) -> tuple[Any, ...]:
    """Parse raw instruction arguments, converting Int to int where appropriate."""
    return tuple(int(arg) if isinstance(arg, Int) else arg for arg in args)


def parse_instruction(
    action: SingleAction,
    state: ScheduleState,
    max_depth: int = 100,
) -> tuple[Instruction[Any], Time | None, float | None]:
    """Parse a single action action into an Instruction.

    Parameters
    ----------
    action : SingleAction
        The action to parse, which can be either a BAction or a CAction,
        following the formats defined below:
        - BAction: (time, instruction_name, *args)
        - CAction: (instruction_name, *args)

    state: ScheduleState
        The current state of the schedule, used for validating the instruction.

    max_depth: int, optional
        The maximum number of times to resolve the instruction to reach a fixed point.
        Default is 100.

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


    Raises
    ------
    ValueError
        If the action is not a valid BAction or CAction, or if the instruction
        name is not registered.

    RuntimeError
        If the instruction fails to reach a fixed point after the specified
        number of resolutions (max_depth).

    Notes
    -----
    Priority is currently not supported in the instruction format.
    Future versions may include priority as an optional argument in the instruction.

    """
    time: Time | None = None

    if not action:
        raise ValueError(
            "Action must have at least one element: (instruction_name, *args), "
            f"but got {action}."
        )

    if isinstance(action[0], Int):
        if len(action) < 2:
            raise ValueError(
                "BAction must have at least two elements: (time, instruction_name, *args), "
                f"but got {action}."
            )

        s_args, spec, *spec_args = cast("BAction", action)
        time = Time(s_args)

    else:
        spec, *spec_args = cast("CAction", action)

    args = _parse_args(spec_args)

    if isinstance(spec, str):
        instruction_cls = instructions.get(spec)

        if instruction_cls is None:
            raise ValueError(
                f"Instruction '{spec}' is not defined, choose one of the "
                f"following: {', '.join(instructions)}."
            )

        spec = instruction_cls

    instruction = spec(*args)
    for _ in range(max_depth):
        validated_instruction = instruction.resolve(state)

        if instruction is validated_instruction:
            break

        instruction = validated_instruction

    else:
        raise RuntimeError(
            f"Instruction {instruction} failed to reach a fixed point after "
            f"{max_depth} resolutions."
        )

    return instruction, time, None

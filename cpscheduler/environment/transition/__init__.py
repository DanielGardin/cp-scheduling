"""Transitions of the scheduling MDP: clock, placement and candidate rule."""

__all__ = [
    "ActionMask",
    "ActionType",
    "AvailableCandidates",
    "CandidateRule",
    "Clock",
    "CompleteInstruction",
    "ConflictSetCandidates",
    "EarliestStartCandidates",
    "EarliestStartPlacement",
    "EventClock",
    "ExecuteInstruction",
    "GlobalClock",
    "HaltInstruction",
    "Instruction",
    "NoClock",
    "NoopInstruction",
    "Placement",
    "RejectInstruction",
    "SingleAction",
    "SkipInstruction",
    "SubmitInstruction",
    "Transition",
    "UnitClock",
    "UnlockedCandidates",
    "is_single_action",
    "parse_instruction",
    "register_instruction",
    "register_transition",
]

from . import presets as presets
from .actions import (
    ActionType,
    Instruction,
    SingleAction,
    is_single_action,
    parse_instruction,
    register_instruction,
)
from .candidates import (
    AvailableCandidates,
    ConflictSetCandidates,
    EarliestStartCandidates,
    UnlockedCandidates,
)
from .clocks import EventClock, NoClock, UnitClock
from .core import (
    ActionMask,
    CandidateRule,
    Clock,
    GlobalClock,
    Placement,
    Transition,
    register_transition,
)
from .instructions import (
    CompleteInstruction,
    ExecuteInstruction,
    HaltInstruction,
    NoopInstruction,
    RejectInstruction,
    SkipInstruction,
    SubmitInstruction,
)
from .placements import EarliestStartPlacement

register_instruction(ExecuteInstruction, "execute", Clock)
register_instruction(SubmitInstruction, "submit", Clock)
register_instruction(RejectInstruction, "reject", Clock)
register_instruction(HaltInstruction, "halt", Clock)
register_instruction(NoopInstruction, "noop", GlobalClock)
register_instruction(CompleteInstruction, "complete", GlobalClock)

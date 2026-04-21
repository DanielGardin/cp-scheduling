__all__ = [
    "SimulationEvent",
    "Schedule",
    "SingleInstruction",
    "ActionType",
    "is_single_action",
    "parse_instruction",
    "instructions",
    # Events
    "ExecuteEvent",
    "SubmitEvent",
    "PauseEvent",
    "ResumeEvent",
    "CheckpointEvent",
    "InterruptEvent",
    "CompleteEvent",
    "AdvanceTimeEvent",
]

from cpscheduler.environment.des.actions import (
    SingleInstruction,
    ActionType,
    is_single_action,
    parse_instruction,
)

from cpscheduler.environment.des.events import (
    AdvanceTimeEvent,
    CheckpointEvent,
    CompleteEvent,
    ExecuteEvent,
    InterruptEvent,
    PauseEvent,
    ResumeEvent,
    SubmitEvent,
)
from cpscheduler.environment.des.base import (
    SimulationEvent,
    Schedule,
    instructions,
    register_instruction,
)

register_instruction(ExecuteEvent, "execute")
register_instruction(SubmitEvent, "submit")
register_instruction(PauseEvent, "pause")
register_instruction(ResumeEvent, "resume")
register_instruction(CheckpointEvent, "noop")
register_instruction(CompleteEvent, "complete")
register_instruction(AdvanceTimeEvent, "advance")

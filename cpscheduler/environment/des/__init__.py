"""Module for discrete event simulation (DES) components of the environment.

This module contains the core components for the discrete event simulation (DES)
of the environment, including the event types, the schedule, and the instruction
parsing utilities.
"""

__all__ = [  # noqa: RUF022
    # Parsing / utilities
    "instructions",
    "is_single_action",
    "parse_instruction",
    # Core types
    "ActionType",
    "Schedule",
    "SingleInstruction",
    "SimulationEvent",
    # Events
    "AdvanceTimeEvent",
    "CheckpointEvent",
    "CompleteEvent",
    "ExecuteEvent",
    "PauseEvent",
    "ResumeEvent",
    "SubmitEvent",
]

from cpscheduler.environment.des.actions import (
    ActionType,
    SingleInstruction,
    is_single_action,
    parse_instruction,
)
from cpscheduler.environment.des.base import (
    SimulationEvent,
    instructions,
    register_instruction,
)
from cpscheduler.environment.des.events import (
    AdvanceTimeEvent,
    CheckpointEvent,
    CompleteEvent,
    ExecuteEvent,
    PauseEvent,
    ResumeEvent,
    SubmitEvent,
)
from cpscheduler.environment.des.schedule import Schedule

register_instruction(ExecuteEvent, "execute")
register_instruction(SubmitEvent, "submit")
register_instruction(PauseEvent, "pause")
register_instruction(ResumeEvent, "resume")
register_instruction(CheckpointEvent, "noop")
register_instruction(CompleteEvent, "complete")
register_instruction(AdvanceTimeEvent, "advance")

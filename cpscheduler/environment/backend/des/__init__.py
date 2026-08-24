"""DES backend module."""

__all__ = [
    "CheckpointEvent",
    "DESBackend",
    "ExecuteEvent",
    "HaltEvent",
    "SkipEvent",
    "SubmitEvent",
]


from cpscheduler.environment.backend.actions import register_instruction

from .des import DESBackend
from .events import (
    CheckpointEvent,
    CompleteEvent,
    ExecuteEvent,
    HaltEvent,
    NOOPEvent,
    SkipEvent,
    SubmitEvent,
)

register_instruction(ExecuteEvent, "execute", "des")
register_instruction(SubmitEvent, "submit", "des")
register_instruction(HaltEvent, "halt", "des")
register_instruction(NOOPEvent, "noop", "des")
register_instruction(CompleteEvent, "complete", "des")

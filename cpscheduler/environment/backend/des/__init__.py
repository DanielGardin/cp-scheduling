"""DES backend module."""

__all__ = [
    "CheckpointEvent",
    "DESBackend",
    "ExecuteEvent",
    "SkipEvent",
    "SubmitEvent",
]


from cpscheduler.environment.backend.actions import register_instruction

from .des import DESBackend
from .events import (
    CheckpointEvent,
    ExecuteEvent,
    SkipEvent,
    SubmitEvent,
)

register_instruction(ExecuteEvent, "execute", "des")
register_instruction(SubmitEvent, "submit", "des")

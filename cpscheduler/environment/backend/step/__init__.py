"""Time-stepped backend module."""

__all__ = ["ExecuteInstruction", "StepBackend"]

from cpscheduler.environment.backend.actions import register_instruction

from .instructions import ExecuteInstruction
from .step import StepBackend

register_instruction(ExecuteInstruction, "execute", "step")

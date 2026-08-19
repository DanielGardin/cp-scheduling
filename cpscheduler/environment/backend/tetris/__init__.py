"""Tetris backend module."""

__all__ = ["ExecuteInstruction", "TetrisBackend"]

from cpscheduler.environment.backend.actions import register_instruction

from .instructions import ExecuteInstruction
from .tetris import TetrisBackend

register_instruction(ExecuteInstruction, "execute", "tetris")

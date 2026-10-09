"""Tetris backend module."""

__all__ = ["ExecuteInstruction", "GTTetrisBackend", "TetrisBackend"]

from cpscheduler.environment.backend.actions import register_instruction

from .gt_tetris import GTTetrisBackend
from .instructions import ExecuteInstruction
from .tetris import TetrisBackend

register_instruction(ExecuteInstruction, "execute", "tetris")
register_instruction(ExecuteInstruction, "execute", "gt_tetris")

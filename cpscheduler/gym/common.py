"""Common constants and methods used in the gymnasium wrapper."""

from gymnasium.spaces import Box, OneOf, Sequence, Text, Tuple
from numpy import int64

from cpscheduler.environment.constants import MAX_TIME

InstructionSpace = Text(max_length=10)
IntSpace = Box(low=0, high=int(MAX_TIME), shape=(), dtype=int64)

SingleInstructionSpace = OneOf(
    [
        Tuple([InstructionSpace]),
        Tuple([InstructionSpace, IntSpace]),
        Tuple([InstructionSpace, IntSpace, IntSpace]),
        Tuple([InstructionSpace, IntSpace, IntSpace, IntSpace]),
        Tuple([IntSpace, InstructionSpace]),
        Tuple([IntSpace, InstructionSpace, IntSpace]),
        Tuple([IntSpace, InstructionSpace, IntSpace, IntSpace]),
        Tuple([IntSpace, InstructionSpace, IntSpace, IntSpace, IntSpace]),
    ]
)

ActionSpace = OneOf(
    [
        SingleInstructionSpace,
        Sequence(SingleInstructionSpace, stack=True),
    ]
)

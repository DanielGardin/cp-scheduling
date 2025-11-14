from typing import Any, overload
from numpy import int64

from gymnasium.spaces import Tuple, Text, Box, OneOf, Sequence

from cpscheduler.environment._common import MAX_INT, Options, InstanceConfig

InstructionSpace = Text(max_length=10)
IntSpace = Box(low=0, high=int(MAX_INT), shape=(), dtype=int64)

SingleActionSpace = OneOf(
    [
        Tuple([InstructionSpace]),
        Tuple([InstructionSpace, IntSpace]),
        Tuple([InstructionSpace, IntSpace, IntSpace]),
        Tuple([InstructionSpace, IntSpace, IntSpace, IntSpace]),
    ]
)

ActionSpace = Sequence(SingleActionSpace, stack=True)

def get_instance_config(options: Options) -> InstanceConfig | None:
    "Construct an instance configuration from the options."
    if not options:
        return None
    
    instance_config: InstanceConfig = {}

    if "instance" in options:
        instance_config["instance"] = options["instance"]

    return instance_config

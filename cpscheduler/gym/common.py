from numpy import int64

from gymnasium.spaces import Tuple, Text, Box, OneOf, Sequence

from cpscheduler.environment.constants import MAX_TIME
from cpscheduler.environment.protocols import Options, InstanceConfig

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

ActionSpace = Sequence(SingleInstructionSpace, stack=True)


def get_instance_config(options: Options) -> InstanceConfig | None:
    "Construct an instance configuration from the options."
    if not options:
        return None

    instance_config: InstanceConfig = {}

    if "instance" in options:
        instance_config["instance"] = options["instance"]

    return instance_config

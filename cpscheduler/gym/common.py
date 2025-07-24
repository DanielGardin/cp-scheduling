from typing import Any, overload
from numpy import int64

from gymnasium.spaces import Tuple, Text, Box, OneOf, Sequence

from cpscheduler.environment._common import MAX_INT, InstanceConfig

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

Options = dict[str, Any] | InstanceConfig | None


@overload
def get_instance_config(options: None) -> None: ...


@overload
def get_instance_config(options: dict[str, Any]) -> InstanceConfig | None: ...


@overload
def get_instance_config(options: InstanceConfig) -> InstanceConfig: ...


def get_instance_config(options: Options) -> InstanceConfig | None:
    "Construct an instance configuration from the options."
    if not options:
        return None

    instance_config: InstanceConfig = {}

    if "instance" in options:
        instance_config["instance"] = options["instance"]

    if "processing_times" in options:
        instance_config["processing_times"] = options["processing_times"]

    if "job_instance" in options:
        instance_config["job_instance"] = options["job_instance"]

    if "job_feature" in options:
        instance_config["job_feature"] = options["job_feature"]

    return instance_config

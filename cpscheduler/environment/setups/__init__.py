"""Module for defining scheduling setups.

Setups are alpha comonents in a scheduling environment, they define the machine
topology, basic constraints and how tasks interact with the machines.

You can define your own setups by subclassing the `ScheduleSetup` class and
implementing the required methods.
"""

__all__ = [
    "FlexibleJobShopSetup",
    "FlowShopSetup",
    "IdenticalParallelMachineSetup",
    "JobShopSetup",
    "OpenShopSetup",
    "ScheduleSetup",
    "SingleMachineSetup",
    "UniformParallelMachineSetup",
    "UnrelatedParallelMachineSetup",
    "setups",
]

from .base import ScheduleSetup, setups
from .parallel import (
    IdenticalParallelMachineSetup,
    SingleMachineSetup,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
)
from .shop import (
    FlexibleJobShopSetup,
    FlowShopSetup,
    JobShopSetup,
    OpenShopSetup,
)

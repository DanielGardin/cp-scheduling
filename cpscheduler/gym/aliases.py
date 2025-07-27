"""
    aliases.py

This module contains environment aliases to simplify the creation of scheduling environments.
Instead of importing the `SchedulingEnv` class directly and working with the available building
blocks like `JobShopSetup`, `Makespan`, etc. from the `cpscheduler.environment` module, users
can import ready-to-use environment.

It is meant to be imported using the `make` function provided by gymnasium
"""

from typing_extensions import Unpack

from cpscheduler.environment._common import InstanceConfig
from cpscheduler.environment.schedule_setup import JobShopSetup
from cpscheduler.environment.objectives import Makespan

from .env import SchedulingEnvGym


def make_jobshop(**instance_config: Unpack[InstanceConfig]) -> SchedulingEnvGym:
    env = SchedulingEnvGym(
        JobShopSetup(),
        objective=Makespan(),
        instance_config=instance_config,
    )

    return env

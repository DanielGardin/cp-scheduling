"""
    aliases.py

This module contains environment aliases to simplify the creation of scheduling environments.
Instead of importing the `SchedulingEnv` class directly and working with the available building
blocks like `JobShopSetup`, `Makespan`, etc. from the `cpscheduler.environment` module, users
can import ready-to-use environment.

It is meant to be imported using the `make` function provided by gymnasium
"""

from cpscheduler.environment._common import InstanceTypes
from cpscheduler.environment.schedule_setup import JobShopSetup
from cpscheduler.environment.objectives import Makespan

from .env import SchedulingEnvGym


def make_jobshop(instance: InstanceTypes) -> SchedulingEnvGym:
    env = SchedulingEnvGym(
        JobShopSetup(),
        objective=Makespan(),
        instance=instance,
    )

    return env

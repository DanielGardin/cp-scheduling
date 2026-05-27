"""
    aliases.py

This module contains environment aliases to simplify the creation of scheduling environments.
Instead of importing the `SchedulingEnv` class directly and working with the available building
blocks like `JobShopSetup`, `Makespan`, etc. from the `cpscheduler.environment` module, users
can import ready-to-use environment.

It is meant to be imported using the `make` function provided by gymnasium
"""

from collections.abc import Mapping
from typing import Any

from cpscheduler.environment.objectives import Makespan
from cpscheduler.environment.setups import JobShopSetup
from cpscheduler.environment.utils.protocols import InstanceTypes, Metric

from .env import SchedulingEnvGym


def make_jobshop(
    instance: InstanceTypes | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
) -> SchedulingEnvGym:
    return SchedulingEnvGym(
        JobShopSetup(), objective=Makespan(), instance=instance, metrics=metrics
    )

from typing import Any
from collections.abc import Callable
from typing_extensions import Unpack

from functools import singledispatch

from cpscheduler.environment._common import InstanceConfig
from cpscheduler.environment.schedule_setup import (
    ScheduleSetup,
    SingleMachineSetup,
    IdenticalParallelMachineSetup,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
    JobShopSetup,
    OpenShopSetup,
)

from ._common import InstanceGeneratorConfig, get_n_jobs, get_processing_times


@singledispatch
def generate_base_instance(
    setup: ScheduleSetup, configs: InstanceGeneratorConfig
) -> InstanceConfig:
    raise NotImplementedError(f"Setup {setup} not implemented for PuLP.")


@generate_base_instance.register
def _(
    setup: SingleMachineSetup, **kwargs: Unpack[InstanceGeneratorConfig]
) -> InstanceConfig:
    n_tasks = get_n_jobs(kwargs)

    instance: InstanceConfig = {}

    processing_times = get_processing_times(
        n_tasks, kwargs.get("processing_time_fn", 1), kwargs
    )

    return instance

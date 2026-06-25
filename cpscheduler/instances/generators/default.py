"""Default sampler implementations for features."""

from typing import Any

from cpscheduler.instances.distributions import (
    PoissonProcess,
    Sampler,
    UniformInt,
    UniformMachineEligibility,
)

DEFAULT_SAMPLERS: dict[str, Sampler[Any]] = {
    "release_times": PoissonProcess(rate=1.0),
    "processing_time": UniformInt(0, 100),
    "weights": UniformInt(1, 10),
    "machine_eligibility": UniformMachineEligibility(p=0.5),
}

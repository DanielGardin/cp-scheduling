"""Jobshop scheduling benchmarks."""

from collections.abc import Mapping
from typing import Any

from cpscheduler.instances.distributions import (
    DeterministicJobAssignment,
    JobPartitionProcess,
    Range,
    Sampler,
    UniformInt,
)
from cpscheduler.instances.generators.benchmarks.benchmarks import Benchmark


@Benchmark.register("partitioned_job_shop")
def partitioned_job_shop_fn(low: int, high: int) -> Mapping[str, Sampler[Any]]:
    """Generate a retangular jobshop instance (n_jobs, n_machines)."""
    return {
        "processing_time": UniformInt(low, high),
        "machine": JobPartitionProcess(Range("n_machines"), shuffle_tasks=True),
        "operation": JobPartitionProcess(Range("n_tasks")),
        "job": DeterministicJobAssignment(),
    }


@Benchmark.register("taillard")
def taillard_fn() -> Mapping[str, Sampler[Any]]:
    """Taillard benchmark configuration for retangular jobshop instances."""
    return partitioned_job_shop_fn(low=1, high=99)

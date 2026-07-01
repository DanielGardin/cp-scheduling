"""Pre-defined environment configurations with name aliases.

This module contains environment aliases to simplify the creation of scheduling environments.
Instead of importing the `SchedulingEnv` class directly and working with the available building
blocks like `JobShopSetup`, `Makespan`, etc. from the `cpscheduler.environment` module, users
can import ready-to-use environment.

It is meant to be imported using the `make` function provided by gymnasium
"""

from collections.abc import Iterable, Mapping
from typing import Any

from cpscheduler.environment.constraints import (
    MachineBreakdownConstraint,
    MachineEligibilityConstraint,
    PrecedenceConstraint,
    ReleaseDateConstraint,
)
from cpscheduler.environment.objectives import (
    Makespan,
    TotalCompletionTime,
    TotalTardiness,
    WeightedCompletionTime,
    WeightedTardiness,
)
from cpscheduler.environment.observation import Observation
from cpscheduler.environment.setups import (
    FlexibleJobShopSetup,
    FlowShopSetup,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    OpenShopSetup,
    SingleMachineSetup,
    UnrelatedParallelMachineSetup,
)
from cpscheduler.environment.tracer import Tracer
from cpscheduler.environment.utils.protocols import InstanceTypes, Metric
from cpscheduler.gym.env import SchedulingEnvGym


def make_jobshop(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Jm||Cmax — Job shop makespan."""
    return SchedulingEnvGym(
        JobShopSetup(),
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_jobshop_dynamic(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Jm|rj|Cmax — Dynamic job shop with release dates."""
    return SchedulingEnvGym(
        JobShopSetup(),
        constraints=[ReleaseDateConstraint()],
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_jobshop_tardiness(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Jm|rj|ΣTj — Dynamic job shop with total tardiness."""
    return SchedulingEnvGym(
        JobShopSetup(),
        constraints=[ReleaseDateConstraint()],
        objective=TotalTardiness(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_jobshop_breakdown(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Jm|brkdwn|Cmax — Job shop with machine breakdowns."""
    return SchedulingEnvGym(
        JobShopSetup(),
        constraints=[MachineBreakdownConstraint()],
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_flexible_jobshop(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """FJm||Cmax — Flexible job shop makespan."""
    return SchedulingEnvGym(
        FlexibleJobShopSetup(),
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_flexible_jobshop_dynamic(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """FJm|rj|Cmax — Dynamic flexible job shop with release dates."""
    return SchedulingEnvGym(
        FlexibleJobShopSetup(),
        constraints=[ReleaseDateConstraint()],
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_flowshop(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Fm||Cmax — Flow shop makespan."""
    return SchedulingEnvGym(
        FlowShopSetup(),
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_openshop(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Om||Cmax — Open shop makespan."""
    return SchedulingEnvGym(
        OpenShopSetup(),
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_parallel_makespan(
    n_machines: int,
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Pm||Cmax — Parallel machines makespan."""
    return SchedulingEnvGym(
        IdenticalParallelMachineSetup(n_machines=n_machines),
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_parallel_total_completion(
    n_machines: int,
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Pm||ΣCj — Parallel machines total completion time."""
    return SchedulingEnvGym(
        IdenticalParallelMachineSetup(n_machines=n_machines),
        objective=TotalCompletionTime(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_parallel_weighted_completion(
    n_machines: int,
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Pm||Σw_jCj — Parallel machines weighted completion time."""
    return SchedulingEnvGym(
        IdenticalParallelMachineSetup(n_machines=n_machines),
        objective=WeightedCompletionTime(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_parallel_dynamic_makespan(
    n_machines: int,
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Pm|rj|Cmax — Parallel machines with release dates, makespan."""
    return SchedulingEnvGym(
        IdenticalParallelMachineSetup(n_machines=n_machines),
        constraints=[ReleaseDateConstraint()],
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_parallel_eligibility_makespan(
    n_machines: int,
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Pm|rj,M_j|Cmax — Parallel machines with release dates and machine eligibility."""
    return SchedulingEnvGym(
        IdenticalParallelMachineSetup(n_machines=n_machines),
        constraints=[ReleaseDateConstraint(), MachineEligibilityConstraint()],
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_unrelated_makespan(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Rm||Cmax — Unrelated parallel machines makespan."""
    return SchedulingEnvGym(
        UnrelatedParallelMachineSetup(),
        objective=Makespan(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_unrelated_weighted_completion(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """Rm||Σw_jCj — Unrelated parallel machines weighted completion time."""
    return SchedulingEnvGym(
        UnrelatedParallelMachineSetup(),
        objective=WeightedCompletionTime(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_single_total_tardiness(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """1||ΣTj — Single machine total tardiness."""
    return SchedulingEnvGym(
        SingleMachineSetup(),
        objective=TotalTardiness(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_single_weighted_tardiness(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """1||Σw_jTj — Single machine weighted total tardiness."""
    return SchedulingEnvGym(
        SingleMachineSetup(),
        objective=WeightedTardiness(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_single_dynamic_tardiness(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """1|rj|ΣTj — Single machine with release dates, total tardiness."""
    return SchedulingEnvGym(
        SingleMachineSetup(),
        constraints=[ReleaseDateConstraint()],
        objective=TotalTardiness(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )


def make_single_precedence_completion(
    instance: InstanceTypes | None = None,
    observation: Observation | None = None,
    metrics: Mapping[str, Metric[Any]] | None = None,
    tracers: Iterable[Tracer] | None = None,
) -> SchedulingEnvGym:
    """1|prec|ΣCj — Single machine with precedence constraints, total completion time."""
    return SchedulingEnvGym(
        SingleMachineSetup(),
        constraints=[PrecedenceConstraint()],
        objective=TotalCompletionTime(),
        observation=observation,
        instance=instance,
        metrics=metrics,
        tracers=tracers,
    )

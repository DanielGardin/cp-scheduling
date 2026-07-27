"""Logic for inferring a adequate sampler from feature metadata."""

from typing import Any

from cpscheduler.environment.constants import MAX_TIME
from cpscheduler.environment.instance import FeatureMetadata
from cpscheduler.instances.distributions import (
    Bernoulli,
    Categorical,
    Dirichlet,
    Exponential,
    Normal,
    Poisson,
    Range,
    Sampler,
    Shuffled,
    Uniform,
    UniformInt,
)


def infer_sampler(
    metadata: FeatureMetadata, symbols: dict[str, Any]
) -> Sampler[Any]:
    """Infer a default sampler from feature metadata."""
    raw_shape = metadata.raw_shape
    shape = metadata.resolve_shape(**symbols)

    match metadata.value_type:
        case "binary":
            return Bernoulli(p=0.5)

        case "continuous":
            return Normal(mean=0.0, stdev=1.0)

        case "normalized":
            return Uniform(low=0.0, high=1.0)

        case "probability":
            if shape and len(shape) > 0:
                last = shape[-1]

                if last is not None:
                    return Dirichlet(alpha=[1.0] * last)

            return Uniform(low=0.0, high=1.0)

        case "cost":
            return Exponential(scale=1.0)

        case "discrete":
            return UniformInt(low=0, high=100)

        case "count":
            return Poisson(rate=10.0)

        case "duration":
            return Poisson(rate=10.0)

        case "time":
            return Exponential(scale=10.0)

        case "order":
            if raw_shape and raw_shape[-1] is not None:
                return Shuffled(Range(raw_shape[-1]))

            return Range()

        case "task_id":
            return UniformInt(
                low=0,
                high=symbols["n_tasks"] - 1,
            )

        case "job_id":
            return UniformInt(
                low=0,
                high=symbols["n_jobs"] - 1,
            )

        case "machine_id":
            return UniformInt(
                low=0,
                high=symbols["n_machines"] - 1,
            )

        case "id":
            return UniformInt(low=-MAX_TIME, high=MAX_TIME)

        case "categorical":
            n_categories = metadata.n_categories

            if n_categories is not None:
                return Categorical(n_categories=n_categories)

            raise ValueError("Cannot infer the number of categories.")

        case "unknown":
            raise ValueError("Cannot infer a sampler for unknown value type.")

    raise AssertionError("Unreachable.")

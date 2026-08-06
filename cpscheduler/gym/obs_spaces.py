"""Module for converting ObservationSpecs to Gymnasium spaces."""

from functools import singledispatch
from typing import Any, cast

import numpy as np
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    Sequence,
    Space,
    Tuple,
)
from typing_extensions import Never

from cpscheduler.environment.constants import MAX_TIME
from cpscheduler.environment.instance.metadata import ValueType
from cpscheduler.environment.specs import (
    DenseViewSpec,
    DictSpec,
    FeatureViewSpec,
    FreeViewSpec,
    GraphSpec,
    ObservationSpec,
    RaggedViewSpec,
    SequenceSpec,
    StackSpec,
)

MAX_INT = float(np.iinfo(np.int32).max)
MAX_FLOAT = float(np.finfo(np.float32).max)
MIN_FLOAT = float(np.finfo(np.float32).min)
MIN_INT = float(np.iinfo(np.int32).min)

BOUNDS: dict[ValueType, tuple[float, float]] = {
    "continuous": (MIN_FLOAT, MAX_FLOAT),
    "discrete": (MIN_INT, MAX_INT),
    "binary": (0.0, 1.0),
    "count": (0.0, MAX_INT),
    "normalized": (0.0, 1.0),
    "probability": (0.0, 1.0),
    "time": (0.0, MAX_TIME),
    "duration": (0.0, MAX_TIME),
    "cost": (MIN_FLOAT, MAX_FLOAT),
    "id": (MIN_INT, MAX_INT),
    "order": (0.0, MAX_INT),
}

_DTYPE = type[np.floating[Any]] | type[np.integer[Any]]
DTYPE_MAPPING: dict[ValueType, _DTYPE] = {
    "continuous": np.float32,
    "discrete": np.int32,
    "binary": np.int8,
    "count": np.int32,
    "normalized": np.float32,
    "probability": np.float32,
    "time": np.int32,
    "duration": np.int32,
    "cost": np.float32,
    "id": np.int32,
    "task_id": np.int32,
    "job_id": np.int32,
    "machine_id": np.int32,
    "order": np.int32,
    "categorical": np.int32,
}


def _resolve_bounds(
    spec: FeatureViewSpec[Any, Any], name: str, symbols: dict[str, int]
) -> tuple[float, float]:
    low, high = spec.low, spec.high

    match spec.value_type:
        case "task_id":
            n_tasks = symbols.get("n_tasks", MAX_INT)
            low = 0.0
            high = float(n_tasks if n_tasks > 0 else MAX_INT)

        case "job_id":
            n_jobs = symbols.get("n_jobs", MAX_INT)
            low = 0.0
            high = float(n_jobs if n_jobs > 0 else MAX_INT)

        case "machine_id":
            n_machines = symbols.get("n_machines", MAX_INT)
            low = 0.0
            high = float(n_machines if n_machines > 0 else MAX_INT)

        case "categorical":
            n_cat = spec.n_categories

            low = 0.0
            high = (
                float(n_cat - 1)
                if n_cat is not None
                else MAX_INT
            )

        case _:
            pass

    concrete_low, concrete_high = BOUNDS.get(spec.value_type, (None, None))

    low = low if low is not None else concrete_low
    high = high if high is not None else concrete_high

    if low is None or high is None:
        raise ValueError(
            f"{name}: Cannot resolve bounds for value_type '{spec.value_type}' "
            f"with low={low} and high={high}."
        )

    if low > high:
        raise ValueError(
            f"{name}: Invalid bounds for value_type '{spec.value_type}': "
            f"low={low} is greater than high={high}."
        )

    return low, high


@singledispatch
def to_gym_space(
    spec: ObservationSpec, name: str, symbols: dict[str, int]
) -> Space[Any]:
    """Convert an ObservationSpec to a Gymnasium space."""
    raise NotImplementedError(
        f"{name}: No gym.Space conversion registered for {type(spec).__name__}."
    )


@to_gym_space.register(DenseViewSpec)
def _(spec: DenseViewSpec[Any], name: str, symbols: dict[str, int]) -> Box:
    if spec.value_type == "unknown":
        raise ValueError(
            f"{name}: Cannot convert DenseViewSpec with unknown value_type to gym.Space."
        )

    low, high = _resolve_bounds(spec, name, symbols)
    return Box(
        low=low,
        high=high,
        shape=spec.resolve_shape(**symbols),
        dtype=DTYPE_MAPPING[spec.value_type],
    )


@to_gym_space.register(RaggedViewSpec)
def _(
    spec: RaggedViewSpec[Any], name: str, symbols: dict[str, int]
) -> Space[Any]:
    if spec.value_type == "unknown":
        raise ValueError(
            f"{name}: Cannot convert RaggedViewSpec with unknown value_type to gym.Space."
        )

    low, high = _resolve_bounds(spec, name, symbols)
    shape = spec.resolve_shape(**symbols)

    dense_shape: list[int] = []

    for dim in reversed(shape):
        if dim is None:
            break

        dense_shape.append(dim)

    dense_shape.reverse()

    space: Space[Any] = Box(
        low=low,
        high=high,
        shape=dense_shape,
        dtype=DTYPE_MAPPING[spec.value_type],
    )

    prefix_len = len(shape) - len(dense_shape)
    for _ in range(prefix_len):
        space = Sequence(space)

    return space


@to_gym_space.register(FreeViewSpec)
def _(spec: FreeViewSpec[Any], name: str, symbols: dict[str, int]) -> Never:
    raise ValueError(
        f"{name}: Cannot convert FreeViewSpec to gym.Space because it has no fixed shape."
    )


@to_gym_space.register
def _(spec: DictSpec, name: str, symbols: dict[str, int]) -> Dict:
    return Dict(
        {key: to_gym_space(field, key, symbols) for key, field in spec.items()}
    )


@to_gym_space.register
def _(spec: SequenceSpec, name: str, symbols: dict[str, int]) -> Space[Any]:
    element_space = to_gym_space(spec.element, name, symbols)

    if spec.length is None:
        return Sequence(element_space)

    return Tuple(
        tuple(element_space for _ in range(spec.length.resolve(**symbols)))
    )


# TODO: Implement StackSpec conversion
@to_gym_space.register
def _(spec: StackSpec, name: str, symbols: dict[str, int]) -> Never:
    raise NotImplementedError(
        f"{name}: No gym.Space conversion registered for StackSpec yet. "
    )


@to_gym_space.register
def _(spec: GraphSpec, name: str, symbols: dict[str, int]) -> Graph:
    node_space = to_gym_space(spec.nodes, name, symbols)
    edge_space = to_gym_space(spec.edges, name, symbols)

    if not isinstance(node_space, Box | Discrete):
        raise ValueError(
            f"GraphSpec nodes must be Box or Discrete, got {type(node_space).__name__}."
        )

    if not isinstance(edge_space, Box | Discrete):
        raise ValueError(
            f"GraphSpec edges must be Box or Discrete, got {type(edge_space).__name__}."
        )

    return Graph(node_space=node_space, edge_space=edge_space)


# Additional registration for extensions of FeatureViewSpec


@to_gym_space.register(FeatureViewSpec)
def _(
    spec: FeatureViewSpec[Any, Any], name: str, symbols: dict[str, int]
) -> Space[Any]:
    try:
        low, high = _resolve_bounds(spec, name, symbols)
        shape = spec.resolve_shape(**symbols)

        if shape is None or any(dim is None for dim in shape):
            raise NotImplementedError(
                f"{name}: {type(spec).__name__} requires a custom gym.Space "
                "conversion because its shape is not fully defined."
            )

        return Box(
            low=low,
            high=high,
            shape=cast("tuple[int, ...]", shape),
            dtype=DTYPE_MAPPING[spec.value_type],
        )

    except Exception as _:
        raise NotImplementedError(
            f"{name}: No gym.Space conversion registered for {type(spec).__name__}."
        ) from None

"""Module for converting ObservationSpecs to Gymnasium spaces."""

from typing import Any, cast

import numpy as np
from gymnasium.spaces import (
    Box,
    Dict,
    Graph,
    Sequence,
    Space,
    Tuple,
)

from cpscheduler.environment.constants import MAX_TIME
from cpscheduler.environment.specs import (
    DictSpec,
    FeatureSpec,
    GraphSpec,
    ObservationSpec,
    SequenceSpec,
    StackSpec,
)
from cpscheduler.environment.specs.feature_spec import SemanticType

MAX_INT = np.iinfo(np.int32).max
MAX_FLOAT = float(np.finfo(np.float32).max)
MIN_FLOAT = float(np.finfo(np.float32).min)
MIN_INT = np.iinfo(np.int32).min


BOUNDS: dict[SemanticType, tuple[float, float]] = {
    "continuous": (MIN_FLOAT, MAX_FLOAT),
    "discrete": (MIN_INT, MAX_INT),
    # "binary": (0.0, 1.0),
    "count": (0.0, MAX_FLOAT),
    "cost": (0.0, MAX_FLOAT),
    "probability": (0.0, 1.0),
    "normalized": (0.0, 1.0),
    "id": (0.0, MAX_INT),
    "order": (0.0, MAX_INT),
    # "categorical": (0.0, n_categories-1),
    # "task": (0.0, n_tasks-1),
    # "job": (0.0, n_jobs-1),
    # "machine": (0.0, n_machines-1),
    "time": (0, int(MAX_TIME)),
    "duration": (0, int(MAX_TIME)),
    "interval": (0, int(MAX_TIME)),  # TODO: separate start/end bounds
    "calendar": (0, int(MAX_TIME)),  # TODO: separate start/end bounds
    # "mask": (0.0, 1.0),
    # "adjacency": (0.0, 1.0),
    # "set": (None, None),
    # "sequence": (None, None),
    # "unknown": (None, None),
}


def _resolve_shape(
    spec: FeatureSpec | StackSpec, symbols: dict[str, int]
) -> tuple[int, ...]:
    shape = spec.resolve_shape(**symbols)

    if shape is None:
        raise ValueError(f"Cannot build space with shapeless spec {spec}.")

    if any(dim is None for dim in shape):
        raise ValueError(
            f"Cannot build space with variadic shape: {shape} from spec {spec}."
        )

    return cast("tuple[int, ...]", shape)


def feature_spec_to_gym_space(
    spec: FeatureSpec, symbols: dict[str, int]
) -> Space[Any]:
    """Convert a FeatureSpec to a corresponding Gymnasium space."""
    low, high = spec.low, spec.high
    semantic = spec.semantic
    shape = _resolve_shape(spec, symbols)

    match semantic:
        case "binary" | "mask":
            return Box(low=0, high=1, shape=shape, dtype=np.int8)

        case "categorical":
            high = (
                spec.n_categories - 1
                if spec.n_categories is not None
                else MAX_INT
            )

            return Box(low=0, high=high, shape=shape, dtype=np.int32)

        # Integer semantics
        case (
            "discrete"
            | "count"
            | "id"
            | "order"
            | "time"
            | "duration"
            | "interval"
            | "calendar"
        ):
            default_low, default_high = BOUNDS[semantic]
            low = low if low is not None else default_low
            high = high if high is not None else default_high
            return Box(low=low, high=high, shape=shape, dtype=np.int32)

        # Float semanticsText,
        case "continuous" | "cost" | "probability" | "normalized":
            default_low, default_high = BOUNDS[semantic]
            low = low if low is not None else default_low
            high = high if high is not None else default_high
            return Box(low=low, high=high, shape=shape, dtype=np.float32)

        case "task":
            return Box(
                low=0, high=symbols["n_tasks"] - 1, shape=shape, dtype=np.int32
            )

        case "job":
            return Box(
                low=0, high=symbols["n_jobs"] - 1, shape=shape, dtype=np.int32
            )

        case "machine":
            return Box(
                low=0,
                high=symbols["n_machines"] - 1,
                shape=shape,
                dtype=np.int32,
            )

        case "adjacency":
            return Box(low=0, high=1, shape=shape, dtype=np.float32)

        case "set":
            # Sets are represented as binary indicator vectors
            raise ValueError("Sets are not supported in this version.")

        case "unknown":
            raise ValueError(
                "Cannot materialize a space for 'unknown' semantic type."
            )


def convert_stack_to_gym_space(spec: StackSpec, symbols: dict[str, int]) -> Box:
    """Convert a StackSpec to a corresponding Gymnasium space."""
    return Box(
        low=MIN_FLOAT,
        high=MAX_FLOAT,
        shape=_resolve_shape(spec, symbols),
        dtype=np.float32,
    )


def convert_spec_to_gym_space(
    spec: ObservationSpec, symbols: dict[str, int]
) -> Space[Any]:
    """Recursively convert an ObservationSpec to a corresponding Gymnasium space."""
    if isinstance(spec, FeatureSpec):
        return feature_spec_to_gym_space(spec, symbols)

    if isinstance(spec, StackSpec):
        return convert_stack_to_gym_space(spec, symbols)

    if isinstance(spec, DictSpec):
        return Dict(
            {
                key: convert_spec_to_gym_space(field_spec, symbols)
                for key, field_spec in spec.items()
            }
        )

    if isinstance(spec, SequenceSpec):
        if spec.length is None:
            return Sequence(convert_spec_to_gym_space(spec.element, symbols))

        element_space = convert_spec_to_gym_space(spec.element, symbols)
        length = spec.length.resolve(**symbols)
        return Tuple(tuple(element_space for _ in range(length)))

    if isinstance(spec, GraphSpec):
        # For graphs, convert node and edge specs
        node_space = convert_stack_to_gym_space(spec.nodes, symbols)
        edge_space = convert_stack_to_gym_space(spec.edges, symbols)
        return Graph(node_space=node_space, edge_space=edge_space)

    raise ValueError(f"Unsupported ObservationSpec type: {type(spec).__name__}")

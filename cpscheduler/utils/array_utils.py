from typing import Any
from collections.abc import Iterable, Iterator

from enum import Enum

from contextlib import contextmanager

NUMPY_AVAILABLE = True
try:
    from numpy.typing import NDArray
    import numpy as np

except ImportError:
    NUMPY_AVAILABLE = False

TORCH_AVAILABLE = True
try:
    import torch

except ImportError:
    TORCH_AVAILABLE = False

from cpscheduler.utils._protocols import ArrayLike, TabularRepresentation
from cpscheduler.utils.list_utils import ListWrapper


@contextmanager
def disable_numpy() -> Iterator[None]:
    """
    Context manager to temporarily disable NumPy functionality in priority dispatching.
    This is mainly used for testing the ListWrapper functionality, without relying on NumPy.

    The reason for this is because we design the library to work with minimal dependencies,
    and NumPy is not a strict requirement for the core functionality.
    """
    global NUMPY_AVAILABLE
    original_numpy_available = NUMPY_AVAILABLE
    NUMPY_AVAILABLE = False
    try:
        yield

    finally:
        NUMPY_AVAILABLE = original_numpy_available


def array_factory(data: Iterable[Any] | ArrayLike) -> ArrayLike:
    array: ArrayLike

    if NUMPY_AVAILABLE:
        array = np.asarray(data)

    else:
        array = ListWrapper(data)

    return array


# The original observation must have shape (*batch, num_tasks, num_features)
# The resulting arraylike after __getitem__ will have shape (*batch, num_tasks)
def wrap_observation(obs: Any) -> TabularRepresentation[ArrayLike]:
    "Wraps the observation into an ArrayLike structure."
    if NUMPY_AVAILABLE and isinstance(obs, np.ndarray):
        return np.moveaxis(obs, -1, 0)

    if TORCH_AVAILABLE and isinstance(obs, torch.Tensor):
        result: ArrayLike = obs.moveaxis(-1, 0)
        return result

    tasks: dict[str, ArrayLike] = {}
    jobs: dict[str, ArrayLike] = {}
    if isinstance(obs, tuple) and len(obs) == 2:
        tasks, jobs = obs

    elif isinstance(obs, dict):
        tasks = obs

    if not tasks:
        raise ValueError("Observation must contain at least task features.")

    new_obs: dict[str, ArrayLike] = {}

    for task_feature in tasks:
        new_obs[task_feature] = array_factory(tasks[task_feature])

    array_jobs_ids = new_obs["job_id"]
    for job_feature in jobs:
        if job_feature == "job_id":
            continue

        new_obs[job_feature] = array_factory(jobs[job_feature])[array_jobs_ids]

    return new_obs


def maximum(x1: Any, x2: Any) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and (
        isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor)
    ):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1)

        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2)

        result = torch.maximum(x1, x2)

    elif NUMPY_AVAILABLE:
        if not isinstance(x1, np.ndarray):
            x1 = np.array(x1)

        if not isinstance(x2, np.ndarray):
            x2 = np.array(x2)

        result = np.maximum(x1, x2)

    else:
        result = ListWrapper.maximum(x1, x2)

    return result


def minimum(x1: Any, x2: Any) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and (
        isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor)
    ):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1)

        if not isinstance(x2, torch.Tensor):
            x2 = torch.tensor(x2)

        result = torch.minimum(x1, x2)

    elif NUMPY_AVAILABLE:
        if not isinstance(x1, np.ndarray):
            x1 = np.array(x1)

        if not isinstance(x2, np.ndarray):
            x2 = np.array(x2)

        result = np.minimum(x1, x2)

    else:
        result = ListWrapper.minimum(x1, x2)

    return result


def argsort(
    x: ArrayLike,
    descending: bool = False,
    stable: bool = False,
    axis: int | None = None,
) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = torch.argsort(x, descending=descending, stable=stable, dim=axis)

    elif NUMPY_AVAILABLE:
        result = np.argsort(
            x if not descending else -x,
            axis=axis,
            kind="stable" if stable else "quicksort",
        )

    else:
        result = ListWrapper.argsort(x, reverse=descending, stable=stable)

    return result


def exp(x: ArrayLike) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = torch.exp(x)

    elif NUMPY_AVAILABLE:
        result = np.exp(x)

    else:
        result = ListWrapper.exp(x)

    return result


def log(x: ArrayLike) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = torch.log(x)

    elif NUMPY_AVAILABLE:
        result = np.log(x)

    else:
        result = ListWrapper.log(x)

    return result


def sqrt(x: ArrayLike) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = torch.sqrt(x)

    elif NUMPY_AVAILABLE:
        result = np.sqrt(x)

    else:
        result = ListWrapper.sqrt(x)

    return result


def where(condition: ArrayLike, x: Any, y: Any) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(condition, torch.Tensor):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        result = torch.where(condition, x, y)

    elif NUMPY_AVAILABLE:
        result = np.where(condition, x, y)

    else:
        result = ListWrapper.where(condition, x, y)

    return result


def array_sum(x: ArrayLike, axis: int | None = None) -> Any:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.sum(x, dim=axis, keepdims=True)

    elif NUMPY_AVAILABLE:
        return np.sum(x, axis=axis)

    elif isinstance(x, ListWrapper):
        return x.sum()

    else:
        return sum(x)


def array_mean(x: ArrayLike, axis: int | None = None) -> Any:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.mean(x, dim=axis, keepdims=True)

    elif NUMPY_AVAILABLE:
        return np.mean(x, axis=axis)

    elif isinstance(x, ListWrapper):
        return x.mean()

    else:
        return sum(x) / len(x) if len(x) > 0 else 0


def array_max(x: ArrayLike, axis: int | None = None) -> Any:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.max(x, dim=axis, keepdim=True)

    elif NUMPY_AVAILABLE:
        return np.max(x, axis=axis)

    elif isinstance(x, ListWrapper):
        return x.max()

    else:
        return max(x) if len(x) > 0 else None


def array_min(x: ArrayLike, axis: int | None = None) -> Any:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.min(x, dim=axis, keepdim=True)

    elif NUMPY_AVAILABLE:
        return np.min(x, axis=axis)

    elif isinstance(x, ListWrapper):
        return x.min()

    else:
        return min(x) if len(x) > 0 else None


def astype(x: ArrayLike, dtype: type) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = x.to(dtype)

    elif NUMPY_AVAILABLE and isinstance(x, np.ndarray):
        result = x.astype(dtype)

    elif isinstance(x, ListWrapper):
        result = x.astype(dtype)

    else:
        raise TypeError(f"Cannot convert {type(x)} to {dtype}.")

    return result

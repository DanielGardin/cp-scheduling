from typing import Any, Literal, Protocol, TypeVar
from collections.abc import Callable
from typing_extensions import TypedDict, NotRequired, Unpack


_T = TypeVar("_T", covariant=True)


class ConfigFn(Protocol[_T]):
    def __call__(self, **kwargs: Unpack["InstanceGeneratorConfig"]) -> _T: ...


class InstanceGeneratorConfig(TypedDict):
    "Additional configurations for instance generation."

    n_jobs: NotRequired[int]
    n_tasks: NotRequired[int]
    processing_time_fn: NotRequired[ConfigFn[int] | int]


def get_n_jobs(kwargs: InstanceGeneratorConfig) -> int:
    if "n_jobs" in kwargs:
        return kwargs["n_jobs"]

    elif "n_tasks" in kwargs:
        return kwargs["n_tasks"]

    raise ValueError("Either 'n_jobs' or 'n_tasks' must be provided.")


def get_processing_times(
    n_tasks: int,
    processing_time_dist: ConfigFn[int] | int,
    configs: InstanceGeneratorConfig,
) -> list[int]:
    if isinstance(processing_time_dist, int):
        return [processing_time_dist] * n_tasks

    return [processing_time_dist(**configs) for _ in range(n_tasks)]

"""Protocol and type helpers for environment configuration.

This module defines lightweight protocols and typed dictionaries used across
the environment to describe metrics, instances, and configuration options.
"""

from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from typing_extensions import TypedDict, TypeVar

if TYPE_CHECKING:
    from cpscheduler.environment.state import ScheduleState

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True, default=Any)


class Metric(Protocol[_T_co]):
    """Metric callable protocol.

    A metric is any callable that receives the current `ScheduleState` and
    returns a value for logging or evaluation.

    Methods
    -------
    __call__(state)
        Compute a metric value from the provided state.

    """

    def __call__(self, state: "ScheduleState") -> _T_co:
        """Compute a metric value from a schedule state.

        Parameters
        ----------
        state : ScheduleState
            Current environment state.

        Returns
        -------
        _T_co
            Metric value.

        """
        ...


@runtime_checkable
class ArrayLike(Protocol[_T]):
    """Minimal protocol for array-like objects.

    This is an extension of Iterable that implements `tolist`.

    Methods
    -------
    __iter__()
        Iterate over the array.

    tolist()
        Return a list equivalent to the array.

    """

    def __iter__(self) -> Iterator[_T]:
        """Iterate over the array."""
        ...

    def tolist(self) -> list[_T]:
        """Return a list equivalent to the array."""
        ...


class DataFrameLike(Protocol):
    """Minimal protocol for data frame-like instances.

    The interface supports column-style access and iteration over column keys.

    Methods
    -------
    __getitem__(key)
        Return a column-like sequence for the given key.

    __iter__()
        Iterate over column keys.

    """

    def __getitem__(self, key: Hashable) -> Any:
        """Return a column-like sequence for the provided key."""
        ...

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over available column keys."""
        ...


Instance_T = DataFrameLike | Mapping[Any, Iterable[Any]]


def prepare_instance(instance: Instance_T) -> dict[str, list[Any]]:
    """Convert instance data into a string-keyed dictionary of lists.

    Parameters
    ----------
    instance : Instance_T
        Instance data mapping column keys to iterables of values.

    Returns
    -------
    dict[str, list[Any]]
        Dictionary with stringified feature names and list values.

    """
    return {str(feature): list(instance[feature]) for feature in instance}


InstanceTypes = Instance_T | tuple[Instance_T, Instance_T]  # Task-instance data


@runtime_checkable
class InstanceGenerator(Protocol):
    """Protocol for components that generate instances from a spec.

    Methods
    -------
    sample(seed=None)
        Sample a new instance (optionally seeded).

    """

    def sample(self, seed: int | None = None) -> InstanceTypes:
        """Sample a new instance.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for reproducibility.

        Returns
        -------
        InstanceTypes
            Sampled instance data.

        """
        ...


InfoType = dict[str, Any]


class InstanceConfig(TypedDict, total=False):
    """Configuration for loading an instance.

    Attributes
    ----------
    instance : InstanceTypes
        Specific instance data to load.

    instance_generator : InstanceGenerator
        Generator used to sample an instance at reset.

    seed : int
        Seed for the instance generator.

    """

    instance: InstanceTypes
    instance_generator: InstanceGenerator
    seed: int


Options = dict[str, Any] | InstanceConfig

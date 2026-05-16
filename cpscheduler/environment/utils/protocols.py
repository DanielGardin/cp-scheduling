from typing import Any, Protocol, TypeVar, runtime_checkable, overload
from collections.abc import Iterator, Mapping, Hashable, Iterable
from typing_extensions import TypedDict

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.instance import ProblemInstance

_T_co = TypeVar("_T_co", covariant=True)


class Metric(Protocol[_T_co]):
    """
    A protocol for metrics that can be used to track and report metrics
    during the scheduling process.
    """

    def __call__(self, state: ScheduleState) -> _T_co: ...


class DataFrameLike(Protocol):
    def __getitem__(self, key: Hashable) -> Any: ...

    def __iter__(self) -> Iterator[Hashable]: ...


Instance_T = DataFrameLike | Mapping[Any, Iterable[Any]]

def prepare_instance(instance: Instance_T) -> dict[str, list[Any]]:
    return {
        str(feature): list(instance[feature])
        for feature in instance
    }


InstanceTypes = (
    ProblemInstance | # Complete specification
    Instance_T | # Task-instance data
    tuple[Instance_T, Instance_T]
)

@runtime_checkable
class InstanceGenerator(Protocol):
    """Protocol for components that can sample a new random instance from an environment spec."""

    @overload
    def sample(self, env: Any, *, seed: int | None = None) -> InstanceTypes: ...

    @overload
    def sample(self, *, seed: int | None = None) -> InstanceTypes: ...

    def sample(
        self, env: Any = None, *, seed: int | None = None
    ) -> InstanceTypes: ...

InfoType = dict[str, Any]


class InstanceConfig(TypedDict, total=False):
    "Configuration for loading an instance."

    instance: InstanceTypes
    instance_generator: InstanceGenerator
    seed: int


Options = dict[str, Any] | InstanceConfig

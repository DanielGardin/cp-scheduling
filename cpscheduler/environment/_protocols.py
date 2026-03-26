from typing import Any, Protocol, TypeVar, runtime_checkable
from collections.abc import Iterator, Mapping, Hashable, Iterable
from typing_extensions import TypedDict, TypeAlias

from cpscheduler.environment.state import ScheduleState

_T_co = TypeVar("_T_co", covariant=True)


class Metric(Protocol[_T_co]):
    """
    A protocol for metrics that can be used to track and report metrics
    during the scheduling process.
    """

    def __call__(self, state: ScheduleState) -> _T_co: ...


@runtime_checkable
class DataFrameLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def columns(self) -> Any: ...

    def __getitem__(self, key: Hashable) -> Any: ...

    def __iter__(self) -> Iterator[Hashable]: ...


InstanceTypes: TypeAlias = DataFrameLike | Mapping[Any, Iterable[Any]]

InfoType: TypeAlias = dict[str, Any]


class InstanceConfig(TypedDict, total=False):
    "Configuration for loading an instance."

    instance: InstanceTypes


Options: TypeAlias = dict[str, Any] | InstanceConfig | None

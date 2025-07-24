from typing import Protocol, TypeVar

from .tasks import Tasks
from .data import SchedulingData

_T = TypeVar("_T", covariant=True)


class Metric(Protocol[_T]):
    """
    A protocol for metrics that can be used to track and report metrics
    during the scheduling process.
    """

    def __call__(self, time: int, tasks: Tasks, data: SchedulingData) -> _T: ...


# Every Objective is a Metric

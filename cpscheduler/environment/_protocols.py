from typing import Protocol, runtime_checkable

from .data import SchedulingData


@runtime_checkable
class ImportableMetric(Protocol):
    def import_data(self, data: SchedulingData) -> None:
        """
        Import data from the SchedulingData object.
        This method is used to initialize the metric with the necessary data.
        """
        pass

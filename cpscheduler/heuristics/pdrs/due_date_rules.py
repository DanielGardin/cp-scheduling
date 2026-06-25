"""Due date-based rules for priority dispatching heuristics."""

from typing_extensions import override

from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.heuristics.pdrs.base import PriorityDispatchingRule


class ModifiedDueDate(PriorityDispatchingRule):
    """Modified Due Date (MDD) heuristic.

    This heuristic prioritizes jobs based on their due dates, or its expected
    completion time when the due date is violated.
    """

    processing_time: str
    due_date: str

    def __init__(
        self,
        processing_time: str = "processing_time",
        due_date: str = "due_date",
        seed: int | None = None,
    ) -> None:
        """Initialize the Modified Due Date heuristic.

        Parameters
        ----------
        processing_time : str, optional
            Feature name for the processing time of each task.
            Default is "processing_time".

        due_date : str, optional
            Feature name for the due date of each task.
            Default is "due_date".

        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        """
        super().__init__(seed)

        self.processing_time = processing_time
        self.due_date = due_date

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        t = obs.time
        due_dates = obs.task[self.due_date]
        processing_times = obs.task[self.processing_time]

        return [
            -max(t + p, d)
            for p, d in zip(processing_times, due_dates, strict=False)
        ]


class WeightedModifiedDueDate(PriorityDispatchingRule):
    """Modified Weighted Due Date (MDD) heuristic.

    This heuristic is a weighted version of the Modified Due Date (MDD) heuristic,
    which prioritizes jobs based on their due dates, or its expected completion
    time when the due date is violated, while also considering the weight of
    each job.
    """

    processing_time: str
    due_date: str
    weight: str

    def __init__(
        self,
        processing_time: str = "processing_time",
        due_date: str = "due_date",
        weight: str = "weight",
        seed: int | None = None,
    ) -> None:
        """Initialize the Weighted Modified Due Date heuristic.

        Parameters
        ----------
        processing_time : str, optional
            Feature name for the processing time of each task.
            Default is "processing_time".

        due_date : str, optional
            Feature name for the due date of each task.
            Default is "due_date".

        weight : str, optional
            Feature name for the weight of each task.
            Default is "weight".

        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        """
        super().__init__(seed)

        self.processing_time = processing_time
        self.due_date = due_date
        self.weight = weight

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        t = obs.time
        due_dates = obs.task[self.due_date]
        processing_times = obs.task[self.processing_time]
        weights = obs.task[self.weight]

        return [
            -max(t + p, d) / w
            for p, d, w in zip(
                processing_times, due_dates, weights, strict=False
            )
        ]


class MinimumSlackTime(PriorityDispatchingRule):
    """
    Minimum Slack Time (MST) heuristic.

    This heuristic prioritizes jobs based on their slack time, which is the
    difference between the due date and the expected completion time of a job.
    """

    due_date: str
    processing_time: str
    release_time: str | None

    def __init__(
        self,
        due_date: str = "due_date",
        processing_time: str = "processing_time",
        release_time: str | None = None,
        seed: int | None = None,
    ):
        """Initialize the Minimum Slack Time heuristic.

        Parameters
        ----------
        due_date : str, optional
            Feature name for the due date of each task.
            Default is "due_date".

        processing_time : str, optional
            Feature name for the processing time of each task.
            Default is "processing_time".

        release_time : str or None, optional
            Feature name for the release time of each task. If None, the release
            time is assumed to be zero for all tasks. Default is None.

        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        """
        super().__init__(seed)

        self.due_date = due_date
        self.processing_time = processing_time
        self.release_time = release_time

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        t = obs.time

        processing_times = obs.task[self.processing_time]
        due_dates = obs.task[self.due_date]

        if self.release_time is None:
            return [
                t + p - d
                for p, d in zip(processing_times, due_dates, strict=False)
            ]

        release_times = obs.task[self.release_time]
        return [
            max(t, r) + p - d
            for p, r, d in zip(
                processing_times, release_times, due_dates, strict=False
            )
        ]

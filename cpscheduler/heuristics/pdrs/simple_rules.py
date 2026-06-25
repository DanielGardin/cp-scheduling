"""Simple priority dispatching rules (PDRs) for scheduling environments."""

from typing_extensions import override

from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.heuristics.pdrs.base import StaticPriorityDispatchingRule


class RandomPriority(StaticPriorityDispatchingRule):
    """Random priority dispatching rule.

    This heuristic assigns a random priority score to each task.
    """

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        return [self._internal_rng.random() for _ in range(obs.n_tasks)]


class ShortestProcessingTime(StaticPriorityDispatchingRule):
    """
    Shortest Processing Time (SPT) heuristic.

    Prioritizes tasks based on their processing time, with shorter tasks
    receiving higher priority.
    """

    processing_time: str

    def __init__(
        self, processing_time: str = "processing_time", seed: int | None = None
    ) -> None:
        """Initialize the Shortest Processing Time heuristic.

        Parameters
        ----------
        processing_time : str, optional
            Feature name for the processing time of each task.
            Default is "processing_time".

        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        """
        super().__init__(seed)

        self.processing_time = processing_time

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        return [float(-p) for p in obs.task[self.processing_time]]


class EarliestDueDate(StaticPriorityDispatchingRule):
    """Earliest Due Date (EDD) heuristic.

    Prioritizes tasks based on their due dates, with earlier due dates receiving
    higher priority.
    """

    due_date: str

    def __init__(
        self, due_date: str = "due_date", seed: int | None = None
    ) -> None:
        """Initialize the Earliest Due Date heuristic.

        Parameters
        ----------
        due_date : str, optional
            Feature name for the due date of each task.
            Default is "due_date".

        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        """
        super().__init__(seed)

        self.due_date = due_date

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        return [float(-d) for d in obs.task[self.due_date]]


class FirstInFirstOut(StaticPriorityDispatchingRule):
    """
    First In First Out (FIFO) heuristic.

    Prioritizes tasks based on their release times, with earlier released tasks
    receiving higher priority.
    """

    release_time: str

    def __init__(
        self, release_time: str = "release_time", seed: int | None = None
    ) -> None:
        """Initialize the First In First Out heuristic.

        Parameters
        ----------
        release_time : str, optional
            Feature name for the release time of each task.
            Default is "release_time".

        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        """
        super().__init__(seed)

        self.release_time = release_time

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        t = obs.time

        return [float(t - r) for r in obs["task"][self.release_time]]

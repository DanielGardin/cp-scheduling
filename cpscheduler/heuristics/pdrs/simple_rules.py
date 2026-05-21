from cpscheduler.environment.observation import DefaultObservation

from cpscheduler.heuristics.pdrs.base import PriorityDispatchingRule

class RandomPriority(PriorityDispatchingRule):
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        return [self._internal_rng.random() for _ in range(obs.n_tasks)]


class ShortestProcessingTime(PriorityDispatchingRule):
    """
    Shortest Processing Time (SPT) heuristic.

    This heuristic selects the job with the shortest processing time as the next job to be scheduled.
    """

    def __init__(
        self, processing_time: str = "processing_time", seed: int | None = None
    ) -> None:
        super().__init__(seed)

        self.processing_time = processing_time

    def priority_score(self, obs: DefaultObservation) -> list[float]:
        return [float(-p) for p in obs.task[self.processing_time]]


class EarliestDueDate(PriorityDispatchingRule):
    """
    Earliest Due Date (EDD) heuristic.

    This heuristic selects the job with the earliest due date as the next job to be scheduled.
    """
    def __init__(
        self, due_date: str = "due_date", seed: int | None = None
    ) -> None:
        super().__init__(seed)

        self.due_date = due_date

    def priority_score(self, obs: DefaultObservation) -> list[float]:
        return [float(-d) for d in obs.task[self.due_date]]


class FirstInFirstOut(PriorityDispatchingRule):
    """
    First In First Out (FIFO) heuristic.

    This heuristic selects the job that has been in the waiting buffer the longest as the next job to be scheduled.
    """
    def __init__(
        self, release_time: str = "release_time", seed: int | None = None
    ) -> None:
        super().__init__(seed)

        self.release_time = release_time

    def priority_score(self, obs: DefaultObservation) -> list[float]:
        t = obs.time

        return [float(t - r) for r in obs["task"][self.release_time]]

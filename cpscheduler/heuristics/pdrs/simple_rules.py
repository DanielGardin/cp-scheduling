from cpscheduler.environment.state import ObsType

from cpscheduler.heuristics.pdrs.base import PriorityDispatchingRule

def get_num_tasks(obs: ObsType) -> int:
    return len(obs[0]["task_id"])

class RandomPriority(PriorityDispatchingRule):
    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        n_tasks = get_num_tasks(obs)
        return [self._internal_rng.random() for _ in range(n_tasks)]


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

    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        return [float(-p) for p in obs[0][self.processing_time]]


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

    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        return [float(-d) for d in obs[0][self.due_date]]


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

    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        t = 0.0 if time is None else float(time)

        return [t - float(r) for r in obs[0][self.release_time]]

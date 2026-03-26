from cpscheduler.environment.state import ObsType

from cpscheduler.heuristics.pdrs.base import PriorityDispatchingRule

class ModifiedDueDate(PriorityDispatchingRule):
    """
    Modified Due Date (MDD) heuristic.
    """

    def __init__(
        self,
        processing_time: str = "processing_time",
        due_date: str = "due_date",
        seed: int | None = None
    ) -> None:
        super().__init__(seed)

        self.processing_time = processing_time
        self.due_date = due_date

    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        t = 0.0 if time is None else float(time)

        due_dates = obs[0][self.due_date]
        processing_times = obs[0][self.processing_time]

        return [
            -max(t + p, d)
            for p, d in zip(processing_times, due_dates)
        ]


class WeightedModifiedDueDate(PriorityDispatchingRule):
    """
    Modified Due Date (MDD) heuristic.
    """

    def __init__(
        self,
        processing_time: str = "processing_time",
        due_date: str = "due_date",
        weight: str = "weight",
        seed: int | None = None
    ) -> None:
        super().__init__(seed)

        self.processing_time = processing_time
        self.due_date = due_date
        self.weight = weight

    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        t = 0.0 if time is None else float(time)

        due_dates = obs[0][self.due_date]
        processing_times = obs[0][self.processing_time]
        weights = obs[0][self.weight]

        return [
            -max(t + p, d) / w
            for p, d, w in zip(processing_times, due_dates, weights)
        ]

class MinimumSlackTime(PriorityDispatchingRule):
    """
    Minimum Slack Time (MST) heuristic.

    This heuristic selects the job with the smallest slack time as the next job to be scheduled.
    """

    def __init__(
        self,
        due_date: str = "due_date",
        processing_time: str = "processing_time",
        release_time: str | None = None,
        seed: int | None = None
    ):
        super().__init__(seed)

        self.due_date = due_date
        self.processing_time = processing_time
        self.release_time = release_time
    
    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        t = 0.0 if time is None else float(time)

        processing_times = obs[0][self.processing_time]
        due_dates = obs[0][self.due_date]

        if self.release_time is None:
            return [
                t + p - d
                for p, d in zip(processing_times, due_dates)
            ]

        release_times = obs[0][self.release_time]
        return [
            max(t, r) + p - d
            for p, r, d in zip(processing_times, release_times, due_dates)
        ]

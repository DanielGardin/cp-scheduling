from typing import SupportsIndex

from cpscheduler.environment.constants import TaskID
from cpscheduler.environment.observation import DefaultObservation

from cpscheduler.heuristics.pdrs.base import PriorityDispatchingRule

# TODO: Generalize these to arbitrary precedence
class MostWorkRemaining(PriorityDispatchingRule):
    """
    Most Work Remaining (MWKR) heuristic.

    This heuristic selects the job with the most work remaining as the next job
    to be scheduled.
    """

    def __init__(
        self,
        processing_time: str = "processing_time",
        operation_label: str = "operation",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)

        self.processing_time = processing_time
        self.operation_label = operation_label

    def priority_score(self, obs: DefaultObservation) -> list[float]:
        task_order = [
            [-1] * len(tasks)
            for tasks in obs.job_tasks
        ]

        operations: list[SupportsIndex] = obs.task[self.operation_label]

        for task_id, job_id in enumerate(obs.job_id):
            op = operations[task_id]
            task_order[job_id][op] = task_id

        work_remaining = [0.0 for _ in range(obs.n_tasks)]

        processing_times = obs.task[self.processing_time]
        for task_ids in task_order:
            cum_work = 0.0
            for task_id in reversed(task_ids):
                cum_work += processing_times[task_id]
                work_remaining[task_id] = cum_work

        return work_remaining


class MostOperationsRemaining(PriorityDispatchingRule):
    """
    Most Operations Remaining (MOPNR) heuristic.

    This heuristic selects the earliest job to be done in the waiting buffer as the next job to be scheduled.
    """

    def __init__(
        self, operation_label: str = "operation", seed: int | None = None
    ) -> None:
        super().__init__(seed)

        self.operation_label = operation_label

    def priority_score(self, obs: DefaultObservation) -> list[float]:
        task_order: dict[TaskID, list[TaskID]] = {}

        job_ids: list[TaskID] = obs.job_id
        operations: list[SupportsIndex] = obs.task[self.operation_label]

        for job_id in job_ids:
            task_order.setdefault(job_id, []).append(-1)

        for task_id, job_id in enumerate(job_ids):
            op = operations[task_id]
            task_order[job_id][op] = task_id

        n_tasks = len(job_ids)
        op_remaining = [0.0 for _ in range(n_tasks)]

        for task_ids in task_order.values():
            for next_ops, task_id in enumerate(reversed(task_ids), start=1):
                op_remaining[task_id] = float(next_ops)

        return op_remaining

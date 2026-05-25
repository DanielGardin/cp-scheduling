from typing import SupportsIndex

from cpscheduler.environment.constants import TaskID
from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.heuristics.pdrs.base import PriorityDispatchingRule


# TODO: Generalize these to arbitrary precedence
class MostWorkRemaining(PriorityDispatchingRule):
    """
    Most Work Remaining (MWKR) heuristic.

    Selects the task belonging to the job with the largest
    remaining cumulative processing time.
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
        operations: list[int] = obs.task[self.operation_label]
        job_ids: list[int] = obs.task["job"]

        processing_times: list[float] = obs.task[self.processing_time]

        max_job_id = max(job_ids, default=-1)

        task_order: list[list[int]] = [[] for _ in range(max_job_id + 1)]

        for task_id, (job_id, operation) in enumerate(
            zip(job_ids, operations, strict=False)
        ):
            tasks = task_order[job_id]

            if len(tasks) <= operation:
                tasks.extend(-1 for _ in range(len(tasks), operation + 1))

            tasks[operation] = task_id

        work_remaining = [0.0] * obs.n_tasks

        for tasks in task_order:
            cumulative = 0.0

            for task_id in reversed(tasks):
                cumulative += float(processing_times[task_id])
                work_remaining[task_id] = cumulative

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

        job_ids: list[TaskID] = obs.task["job"]
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

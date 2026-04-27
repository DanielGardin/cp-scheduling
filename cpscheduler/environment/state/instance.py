from typing import Any

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    MachineID, TaskID, Time,
    EzPickle
)

from cpscheduler.environment.utils import convert_to_list


def check_instance_consistency(instance: dict[str, list[Any]]) -> int:
    "Check if all lists in the instance have the same length."
    it = iter(instance.values())
    try:
        first = len(next(it))

    except StopIteration:
        return 0

    for v in it:
        if len(v) != first:
            raise ValueError(
                "Inconsistent instance, all lists must have the same length, "
                f"but got lengths {first} and {len(v)} for keys "
                f"{list(instance.keys())}."
            )

    return first


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class ProblemInstance(EzPickle):
    __args__ = ("task_instance",)

    job_ids: list[TaskID]
    job_tasks: list[list[TaskID]]

    preemptive: list[bool]
    optional: list[bool]
    processing_times: list[dict[MachineID, Time]]

    task_instance: dict[str, list[Any]]

    n_tasks: int
    n_jobs: int
    n_machines: int

    def __init__(self, task_instance: dict[str, list[Any]]) -> None:
        task_instance = task_instance.copy()

        n_tasks = check_instance_consistency(task_instance)

        job_ids = convert_to_list(
            (
                task_instance["job"]
                if "job" in task_instance
                else range(n_tasks)
            ),
            TaskID,
        )

        n_jobs = max(job_ids) + 1 if job_ids else 0
        job_tasks: list[list[TaskID]] = [[] for _ in range(n_jobs)]
        for task_id, job_id in enumerate(job_ids):
            job_tasks[job_id].append(task_id)

        self.preemptive = [False] * n_tasks
        self.optional = [False] * n_tasks
        self.processing_times = [{} for _ in range(n_tasks)]

        self.job_tasks = job_tasks
        task_instance["job_id"] = job_ids
        task_instance["task_id"] = list(range(n_tasks))

        self.task_instance = task_instance
        self.job_ids = job_ids

        self.n_tasks = n_tasks
        self.n_jobs = n_jobs
        self.n_machines = 0

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, ProblemInstance):
            return False

        return (
            self.job_ids == value.job_ids
            and self.job_tasks == value.job_tasks
            and self.preemptive == value.preemptive
            and self.optional == value.optional
            and self.processing_times == value.processing_times
            and self.task_instance == value.task_instance
            and self.n_tasks == value.n_tasks
            and self.n_jobs == value.n_jobs
            and self.n_machines == value.n_machines
        )

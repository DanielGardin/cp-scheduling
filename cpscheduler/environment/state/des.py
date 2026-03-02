from typing import Any
from collections.abc import KeysView

from cpscheduler.environment.constants import (
    MachineID,
    TaskID,
    Time,
    MAX_TIME
)

from cpscheduler.utils.list_utils import convert_to_list


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

class ProblemInstance:
    __slots__ = (
        "job_ids",
        "job_tasks",
        "preemptive",
        "optional",
        "processing_times",
        "task_instance",
        "n_tasks",
        "n_jobs",
        "_n_machines",
    )

    job_ids: list[TaskID]
    job_tasks: list[list[TaskID]]

    preemptive: list[bool]
    optional: list[bool]
    processing_times: list[dict[MachineID, Time]]

    task_instance: dict[str, list[Any]]

    n_tasks: int
    n_jobs: int
    _n_machines: int

    def __init__(
        self,
        task_instance: dict[str, list[Any]],
    ) -> None:
        self.task_instance = task_instance.copy()

        self.n_tasks = check_instance_consistency(task_instance)

        job_ids = convert_to_list(
            (
                task_instance["job"]
                if "job" in task_instance
                else range(self.n_tasks)
            ),
            TaskID,
        )

        self.job_ids = job_ids
        self.n_jobs = max(job_ids) + 1 if job_ids else 0

        self.preemptive = [False] * self.n_tasks
        self.optional = [False] * self.n_tasks
        self.processing_times = [{} for _ in range(self.n_tasks)]

        self._n_machines = 0

        self.job_tasks = [[] for _ in range(self.n_jobs)]
        for task_id, job_id in enumerate(job_ids):
            self.job_tasks[job_id].append(task_id)

        self.task_instance["job_id"] = self.job_ids
        self.task_instance["task_id"] = list(range(self.n_tasks))

    @property
    def n_machines(self) -> int:
        if self._n_machines > 0:
            return self._n_machines

        for p_times in self.processing_times:
            for machine_id in p_times:
                if machine_id >= self._n_machines:
                    self._n_machines = machine_id + 1

        return self._n_machines

    def is_preemptive(self, task_id: TaskID) -> bool:
        "Check if a task allows preemption."
        return self.preemptive[task_id]

    def is_optional(self, task_id: TaskID) -> bool:
        "Check if a task is optional."
        return self.optional[task_id]

    def get_processing_time(
        self, task_id: TaskID, machine_id: MachineID
    ) -> Time:
        "Get the processing time for a given task and machine."
        return self.processing_times[task_id].get(machine_id, MAX_TIME)

    def get_machines(self, task_id: TaskID) -> KeysView[MachineID]:
        "Get the set of machines that can process a given task."
        return self.processing_times[task_id].keys()

    def set_preemption(
        self, task_id: TaskID, allow_preemption: bool = True
    ) -> None:
        "Set whether a task allows preemption."
        self.preemptive[task_id] = allow_preemption

    def set_optionality(self, task_id: TaskID, optional: bool = True) -> None:
        "Set whether a task is optional."
        self.optional[task_id] = optional

    def set_processing_time(
        self, task_id: TaskID, machine_id: MachineID, time: Time
    ) -> None:
        "Set the processing time for a given task and machine."
        if time < 0:
            raise ValueError("Processing time cannot be negative.")

        self.processing_times[task_id][machine_id] = time

    def remove_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        "Remove a machine from processing a given task."
        if machine_id in self.processing_times[task_id]:
            del self.processing_times[task_id][machine_id]

    # Dunder methods
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ProblemInstance):
            return False

        return self.task_instance == value.task_instance

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.task_instance,
            self.n_tasks,
            self.n_jobs,
            self._n_machines,
            self.job_ids,
            self.job_tasks,
            self.preemptive,
            self.optional,
            self.processing_times,
        )
        return (self.__class__.__new__, (self.__class__,), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.task_instance,
            self.n_tasks,
            self.n_jobs,
            self._n_machines,
            self.job_ids,
            self.job_tasks,
            self.preemptive,
            self.optional,
            self.processing_times,
        ) = state

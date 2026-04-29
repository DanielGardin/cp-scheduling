from typing import Any, TypeVar, overload

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

_DataType = TypeVar("_DataType")

@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class ProblemInstance(EzPickle):
    job_ids: list[TaskID]
    job_tasks: list[list[TaskID]]

    preemptive: list[bool]
    optional: list[bool]
    processing_times: list[dict[MachineID, Time]]

    task_instance: dict[str, list[Any]]
    job_instance: dict[str, list[Any]]
    global_instance: dict[str, Any]

    n_tasks: int
    n_jobs: int
    n_machines: int

    _frozen: bool

    def __init__(
        self,
        task_instance: dict[str, list[Any]] | None = None,
        job_instance: dict[str, list[Any]] | None = None,
    ) -> None:
        if task_instance is None:
            task_instance = {}

        if job_instance is None:
            job_instance = {}

        n_tasks = check_instance_consistency(task_instance)
        check_instance_consistency(job_instance)

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
        self.job_ids = job_ids

        self.task_instance = task_instance
        self.job_instance = job_instance
        self.global_instance = {}

        self.n_tasks = n_tasks
        self.n_jobs = n_jobs
        self.n_machines = 0

        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _check_mutable(self, origin: str) -> None:
        if self._frozen:
            raise RuntimeError(
                f"{origin}: ProblemInstance is frozen and cannot be modified."
            )

    def set_processing_time(
        self, task_id: TaskID, machine_id: MachineID, time: Time
    ) -> None:
        self._check_mutable("set_processing_time")

        if time < 0:
            raise ValueError("Processing time cannot be negative.")

        if machine_id >= self.n_machines:
            self.n_machines = machine_id + 1

        self.processing_times[task_id][machine_id] = time

    def remove_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        self._check_mutable("remove_machine")

        self.processing_times[task_id].pop(machine_id, None)

    def set_preemption(self, task_id: TaskID, allow_preemption: bool = True) -> None:
        self._check_mutable("set_preemption")

        self.preemptive[task_id] = allow_preemption

    def set_optionality(self, task_id: TaskID, optional: bool = True) -> None:
        self._check_mutable("set_optionality")

        self.optional[task_id] = optional

    @overload
    def register_task_feature(
        self, feature: str, data: list[_DataType]
    ) -> list[_DataType]: ...

    @overload
    def register_task_feature(
        self, feature: str, data: None = None
    ) -> list[Any]: ...

    def register_task_feature(
        self, feature: str, data: list[Any] | None = None
    ) -> list[Any]:
        if data is None:
            if feature not in self.task_instance:
                raise KeyError(
                    f"register_task_feature: '{feature}' is not registered."
                )

            return self.task_instance[feature]

        if feature in self.task_instance:
            if self.task_instance[feature] != data:
                raise ValueError(
                    f"register_task_feature: '{feature}' is already registered "
                    f"with different data."
                )

            return self.task_instance[feature]

        if len(data) != self.n_tasks:
            raise ValueError(
                f"register_task_feature: '{feature}' has length {len(data)}, "
                f"expected {self.n_tasks}."
            )

        self._check_mutable(f"register_task_feature({feature})")
        self.task_instance[feature] = data
        return self.task_instance[feature]

    @overload
    def register_job_feature(
        self, feature: str, data: list[_DataType]
    ) -> list[_DataType]: ...

    @overload
    def register_job_feature(
        self, feature: str, data: None = None
    ) -> list[Any]: ...

    def register_job_feature(
        self, feature: str, data: list[Any] | None = None
    ) -> list[Any]:
        if data is None:
            if feature not in self.job_instance:
                raise KeyError(
                    f"register_job_feature: '{feature}' is not registered."
                )

            return self.job_instance[feature]

        if feature in self.job_instance:
            if self.job_instance[feature] != data:
                raise ValueError(
                    f"register_job_feature: '{feature}' is already registered "
                    f"with different data."
                )

            return self.job_instance[feature]

        if len(data) != self.n_jobs:
            raise ValueError(
                f"register_job_feature: '{feature}' has length {len(data)}, "
                f"expected {self.n_jobs}."
            )

        self._check_mutable(f"register_job_feature({feature})")
        self.job_instance[feature] = data
        return self.job_instance[feature]

    @overload
    def register_global_feature(
        self, feature: str, data: _DataType
    ) -> _DataType: ...

    @overload
    def register_global_feature(
        self, feature: str, data: None = None
    ) -> Any: ...

    def register_global_feature(
        self, feature: str, data: Any | None = None
    ) -> Any:
        if data is None:
            if feature not in self.global_instance:
                raise KeyError(
                    f"register_global_feature: '{feature}' is not registered."
                )

            return self.global_instance[feature]

        if feature in self.global_instance:
            if self.global_instance[feature] != data:
                raise ValueError(
                    f"register_job_feature: '{feature}' is already registered "
                    f"with different data."
                )

            return self.global_instance[feature]

        self._check_mutable(f"register_global_feature({feature})")
        self.global_instance[feature] = data
        return self.global_instance[feature]

    def freeze(self) -> None:
        self._frozen = True

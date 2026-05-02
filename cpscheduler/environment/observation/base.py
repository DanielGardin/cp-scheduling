from typing import Any, Literal, overload
from collections.abc import Iterator
from typing_extensions import Self

from copy import deepcopy

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle, TaskID
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state import ScheduleState

@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Observation(EzPickle):
    """Base observation class, which exposes basic information from the environment.

    This class is the most simple observation, acting as a copy of ProblemInstance,
    designed to be subclassed when a different observation schema is required.

    Attributes:
        n_tasks: Number of tasks in the current instance
        n_jobs: Number of jobs in the current instance
        n_machines: Number of machines in the current instance
        time: Current time

    Properties:
        task: Dictionary containing task-specific data
        job: Dictionary containing job-specific data
        global_: Dictionary containing global data
        available: List of booleans indicating whether each task is available
        status: List of the status of each task
        job_id: List of the respective job of each task
    """

    n_tasks: int
    n_jobs: int
    n_machines: int
    time: int

    job_tasks: list[list[TaskID]]
    available_tasks: set[TaskID]

    _task: dict[str, list[Any]]
    _job: dict[str, list[Any]]
    _global: dict[str, Any] = {}

    def initialize(self, instance: ProblemInstance) -> None:
        self.n_tasks = instance.n_tasks
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
        self.time = 0

        # These copies are shallow, do not modify them during action selection.
        self.job_tasks = instance.job_tasks.copy()
        self._task = instance.task_instance.copy()
        self._job = instance.job_instance.copy()
        self._global = instance.global_instance.copy()

        # dynamic slots pre-allocated so update() never inserts new keys
        self._task["status"] = [0] * instance.n_tasks
        self._task["job_id"] = instance.job_ids.copy()
        self.available_tasks = set()


    def update(self, state: ScheduleState) -> None:
        task = self._task

        task["status"] = state.runtime.status.copy()

        self._state_time = state.time

        self.available_tasks.clear()
        for task_id in state.runtime.unlocked_tasks:
            if state.is_available(task_id):
                self.available_tasks.add(task_id)

    @property
    def task(self) -> dict[str, list[Any]]:
        return self._task

    @property
    def job(self) -> dict[str, list[Any]]:
        return self._job

    @property
    def global_(self) -> dict[str, Any]:
        return self._global

    @property
    def status(self) -> list[int]:
        return self._task["status"]

    @property
    def job_id(self) -> list[int]:
        return self._task["job_id"]


    @overload
    def __getitem__(self, key: Literal[0, 1]) -> dict[str, list[Any]]: ...

    @overload
    def __getitem__(self, key: Literal[2]) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: Literal[0, 1, 2] | str) -> Any:
        if key == 0:
            return self._task

        if key == 1:
            return self._job

        if key == 2:
            return self._global

        if key in self._task:
            return self._task[key]

        if key in self._job:
            return self._job[key]

        if key in self._global:
            return self._global[key]

        if key == "time":
            return self.time

        raise KeyError(f"Key '{key}' not found in task, job, or global observation.")

    def __iter__(self) -> Iterator[dict[str, list[Any]]]:
        yield self._task
        yield self._job
        yield self._global

    def __len__(self) -> int:
        return 3

    def __contains__(self, key: object) -> bool:
        if isinstance(key, int):
            return key in (0, 1, 2)

        if isinstance(key, str):
            return key in self._task or key in self._job or key in self._global or key == "time"

        return False

    def keys(self) -> tuple[str, str, str]:
        return ("task", "job", "global")

    def values(self) -> tuple[dict[str, list[Any]], dict[str, list[Any]], dict[str, Any]]:
        return self._task, self._job, self._global

    def items(self) -> tuple[
        tuple[str, dict[str, list[Any]]],
        tuple[str, dict[str, list[Any]]],
        tuple[str, dict[str, Any]],
    ]:
        return (("task", self._task), ("job", self._job), ("global", self._global))

    def to_tuple(self) -> tuple[dict[str, list[Any]], dict[str, list[Any]], dict[str, Any]]:
        return self._task, self._job, self._global

    def to_dicts(self) -> dict[str, dict[str, list[Any]]]:
        return {"task": self._task, "job": self._job, "global": self._global}

    def clone(self) -> Self:
        return deepcopy(self)

    def __repr__(self) -> str:
        task_keys = ", ".join(sorted(self._task))
        job_keys = ", ".join(sorted(self._job))
        global_keys = ", ".join(sorted(self._global))
        return (
            f"Observation(t={self.time}, "
            f"task=[{task_keys}], job=[{job_keys}], global=[{global_keys}])"
        )

    def __eq__(self, value: Any) -> bool:
        return (
            isinstance(value, Observation)
            and self.time == value.time
            and self._task == value._task
            and self._job == value._job
            and self._global == value._global
        )

    def __hash__(self) -> int:
        return hash((
            self.n_tasks,
            self.n_jobs,
            self.n_machines,
            self.time,
            tuple(self._task["status"]),
            tuple(self.available_tasks),
        ))

from typing import Any, Literal, Generic, overload
from collections.abc import Iterator
from typing_extensions import Self, TypeVar

from copy import deepcopy

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle, MachineID, TaskID, Time
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state import ScheduleState

Serialized_Obs = TypeVar("Serialized_Obs", default=Any)

@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Observation(EzPickle, Generic[Serialized_Obs]):
    """Abstract observation contract for scheduling environments."""

    def initialize(self, instance: ProblemInstance) -> None:
        """Initialize the observation with the scheduling instance."""

    def update(self, state: ScheduleState) -> None:
        """Update the observation from the current stable scheduling state.
        
        This function is called immediatelly before the observation is returned 
        in the `step` and `reset` methods.
        Consider this method as a importer of the most recent 
        """

    def on_task_started(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle a task start event."""

    def on_task_paused(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle a task pause event."""

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle a task completion event."""

    def on_task_machine_infeasible(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle a task-machine infeasibility event."""

    def serialize(self) -> Serialized_Obs:
        """Return a serialized representation of the observation."""
        raise NotImplementedError(
            f"serialize() was not implemented for {type(self).__name__}."
        )

@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class DefaultObservation(Observation[dict[str, Any]]):
    # Global information
    n_tasks: int
    n_jobs: int
    n_machines: int
    time: int

    job_tasks: list[list[TaskID]]
    available_tasks: set[TaskID]
    original_processing_times: list[dict[MachineID, Time]]
    processing_times: list[dict[MachineID, Time]]

    # Internal storage for the namespaces, do not modify directly.
    _task: dict[str, list[Any]]
    _job: dict[str, list[Any]]
    _global: dict[str, Any]

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
        self.original_processing_times = deepcopy(instance.processing_times)

        # dynamic slots pre-allocated so update() never inserts new keys
        self._task["status"] = [0] * instance.n_tasks
        self._task["job_id"] = instance.job_ids.copy()
        self.available_tasks = set()
        self.processing_times = [
            p.copy() for p in self.original_processing_times
        ]

    def update(self, state: ScheduleState) -> None:
        task = self._task

        task["status"][:] = state.runtime.status

        self.time = state.time

        self.available_tasks.clear()
        for task_id in state.runtime.unlocked_tasks:
            if state.is_available(task_id):
                self.available_tasks.add(task_id)

    def on_task_started(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the start of a task on a specific machine."""
        self.processing_times[task_id] = {
            machine_id: state.get_remaining_time(task_id, machine_id)
        }

    def on_task_paused(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the pause of a task on a specific machine."""
        self.processing_times[task_id] = {
            m_id: state.get_remaining_time(task_id, m_id)
            for m_id in self.processing_times[task_id]
        }

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the completion of a task on a specific machine."""
        self.processing_times[task_id].clear()

    def on_task_machine_infeasible(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the loss of feasibility for a task on a specific machine."""
        del self.processing_times[task_id][machine_id]


    @overload
    def __getitem__(self, key: Literal["task", "job"]) -> dict[str, list[Any]]: ...

    @overload
    def __getitem__(self, key: Literal["global"]) -> dict[str, Any]: ...

    def __getitem__(self, key: str) -> Any:
        if key == "task":
            return self._task

        if key == "job":
            return self._job

        if key == "global":
            return self._global

        raise KeyError(
            "Observation namespaces are: "
            "'task', 'job', and 'global'."
        )

    def __contains__(self, key: object) -> bool:
        return (
            isinstance(key, str)
            and key in ("task", "job", "global")
        )

    def __iter__(self) -> Iterator[str]:
        yield "task"
        yield "job"
        yield "global"

    def __len__(self) -> int:
        return 3

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

    def serialize(self) -> dict[str, Any]:
        return {"task": self._task, "job": self._job, "global": self._global}

    def clone(self) -> Self:
        return deepcopy(self)

    def __repr__(self) -> str:
        task_keys = ", ".join(sorted(self._task))
        job_keys = ", ".join(sorted(self._job))
        global_keys = ", ".join(sorted(self._global))

        return (
            f"Observation("
            f"time={self.time}, "
            f"task=[{task_keys}], "
            f"job=[{job_keys}], "
            f"global=[{global_keys}]"
            f")"
        )

    def __eq__(self, value: Any) -> bool:
        return (
            isinstance(value, DefaultObservation)
            and self.time == value.time
            and self._task == value._task
            and self._job == value._job
            and self._global == value._global
            and self.original_processing_times == value.original_processing_times
            and self.processing_times == value.processing_times
        )

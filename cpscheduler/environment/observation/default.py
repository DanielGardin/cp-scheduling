from typing import Any

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import TaskID, Time

from cpscheduler.environment.instance import (
    ProblemInstance,
    TaskFeature,
    JobFeature,
    MachineFeature,
    GlobalFeature,
)

from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.observation.base import Observation


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class DefaultObservation(Observation[dict[str, dict[str, Any]]]):
    """
    Lightweight default observation.

    Static instance features are copied once during initialization.
    Runtime buffers are updated in-place.
    """

    time: Time

    task: dict[str, Any]
    job: dict[str, Any]
    machine: dict[str, Any]
    global_state: dict[str, Any]

    available_tasks: set[TaskID]

    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)

        self.time = 0

        self.available_tasks = set()

        self.task = {}
        self.job = {}
        self.machine = {}
        self.global_state = {}

        for features in instance.features.values():
            if not features:
                continue

            feature = features[0]

            if not feature.loaded:
                continue

            if isinstance(feature, TaskFeature):
                self.task[feature.name] = feature.value

            elif isinstance(feature, JobFeature):
                self.job[feature.name] = feature.value

            elif isinstance(feature, MachineFeature):
                self.machine[feature.name] = feature.value

            elif isinstance(feature, GlobalFeature):
                self.global_state[feature.name] = feature.value

        self.task["status"] = [0] * self.n_tasks
        self.task["available"] = [False] * self.n_tasks

        self.global_state["time"] = 0
        self.global_state["infeasible"] = False

    def update(self, state: ScheduleState) -> None:
        self.time = state.time

        self.global_state["time"] = state.time
        self.global_state["infeasible"] = state.infeasible

        status = self.task["status"]
        status[:] = state.runtime.status

        available = self.task["available"]

        self.available_tasks.clear()

        for task_id in range(self.n_tasks):
            is_available = (
                task_id in state.runtime.unlocked_tasks
                and state.is_available(task_id)
            )

            available[task_id] = is_available

            if is_available:
                self.available_tasks.add(task_id)

    def __getitem__(self, key: str) -> dict[str, Any]:
        if key == "task":
            return self.task

        if key == "job":
            return self.job

        if key == "machine":
            return self.machine

        if key == "global":
            return self.global_state

        raise KeyError(f"Unknown observation scope '{key}'.")

    def serialize(self) -> dict[str, dict[str, Any]]:
        return {
            "task": self.task,
            "job": self.job,
            "machine": self.machine,
            "global": self.global_state,
        }

    def __repr__(self) -> str:
        return (
            f"DefaultObservation("
            f"time={self.time}, "
            f"n_tasks={self.n_tasks}, "
            f"n_jobs={self.n_jobs}, "
            f"n_machines={self.n_machines}"
            f")"
        )

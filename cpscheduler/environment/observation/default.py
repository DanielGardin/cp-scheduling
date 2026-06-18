"""Default observation for scheduling environments."""

from typing import Any, Literal, TypedDict, overload

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import Status, StatusType, TaskID, Time
from cpscheduler.environment.instance import (
    GlobalFeature,
    JobFeature,
    MachineFeature,
    ProblemInstance,
    TaskFeature,
)
from cpscheduler.environment.observation.base import Observation
from cpscheduler.environment.observation.runtime_feature import RuntimeFeature
from cpscheduler.environment.specs import DictSpec, FeatureSpec
from cpscheduler.environment.state import ScheduleState

DefaultObsType = TypedDict(
    "DefaultObsType",
    {
        "task": dict[str, list[Any]],
        "job": dict[str, list[Any]],
        "machine": dict[str, list[Any]],
        "global": dict[str, Any],
    },
)

AWAITING = Status.AWAITING


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class DefaultObservation(Observation[DefaultObsType]):
    """Lightweight default observation.

    This is the default observation returned by the environment when none is
    explicitly selected.
    It provides a simple way of accessing the features defined in every component
    of the environment, without the need of defining a custom observation class.

    The features are organized in four scopes: "task", "job", "machine" and "global",
    which can be accessed as dictionaries of feature name to feature value.

    Because it is too general, it is not recommended for training agents,
    but it can be useful for debugging and testing purposes.
    """

    _exclude_features: set[str]
    _specs: dict[str, FeatureSpec]

    _time: RuntimeFeature[Time]
    _status: RuntimeFeature[list[StatusType]]
    _available: RuntimeFeature[list[bool]]

    task: dict[str, Any]
    job: dict[str, Any]
    machine: dict[str, Any]
    global_state: dict[str, Any]

    available_tasks: set[TaskID]

    def __init__(self, exclude_features: set[str] | None = None) -> None:
        """Initialize the DefaultObservation.

        Parameters
        ----------
        exclude_features: set[str] | None
            A set of feature names to exclude from the observation.
            If None, all features will be included.

        """
        self._exclude_features = exclude_features or set()

        self._status = RuntimeFeature(
            name="status",
            scope="task",
            semantic="categorical",
            data=[],
            n_categories=Status.count(),
            shape=("n_tasks",),
        )

        self._available = RuntimeFeature(
            name="available",
            scope="task",
            semantic="mask",
            data=[],
            shape=("n_tasks",),
        )

        self._time = RuntimeFeature(
            name="time",
            scope="global",
            semantic="time",
            data=0,
            shape=(),
        )

    @property
    def time(self) -> Time:
        """Return the current time in the schedule."""
        return self._time.value

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)
        self._specs = {
            "status": self._status.spec,
            "available": self._available.spec,
            "time": self._time.spec,
        }

        self.available_tasks = set()

        self.task = {
            "status": self._status.value,
            "available": self._available.value,
        }
        self.job = {}
        self.machine = {}
        self.global_state = {
            "time": self._time.value,
        }

        for feat_name, features in instance.features.items():
            if feat_name in self._exclude_features or not features:
                continue

            feature = features[0]

            if not feature.loaded:
                raise ValueError(
                    f"Feature '{feat_name}' is not loaded in the instance."
                )

            self._specs[feat_name] = feature.spec

            if isinstance(feature, TaskFeature):
                self.task[feat_name] = feature.value

            elif isinstance(feature, JobFeature):
                self.job[feat_name] = feature.value

            elif isinstance(feature, MachineFeature):
                self.machine[feat_name] = feature.value

            elif isinstance(feature, GlobalFeature):
                self.global_state[feat_name] = feature.value

    @override
    def reset(self, state: ScheduleState) -> None:
        self._time.value = state.time
        self._status.value.clear()
        self._status.value.extend(state.runtime.status)
        self._available.value.clear()
        self._available.value.extend([False] * self.n_tasks)

    @override
    def update(self, state: ScheduleState) -> None:
        self._time.value = state.time
        self._status.value[:] = state.runtime.status

        available = self._available.value
        for task_id in range(self.n_tasks):
            available[task_id] = False

        available_tasks = state.get_available_tasks()
        for task_id in available_tasks:
            available[task_id] = True
            self.available_tasks.add(task_id)

        self.available_tasks.clear()
        self.available_tasks.update(available_tasks)

    @overload
    def __getitem__(
        self, key: Literal["task", "job", "machine"]
    ) -> dict[str, list[Any]]: ...

    @overload
    def __getitem__(self, key: Literal["global"]) -> dict[str, Any]: ...

    def __getitem__(self, key: str) -> Any:
        """Get the features of the specified scope."""
        if key == "task":
            return self.task

        if key == "job":
            return self.job

        if key == "machine":
            return self.machine

        if key == "global":
            return self.global_state

        raise KeyError(f"Unknown observation scope '{key}'.")

    @override
    def serialize(self) -> DefaultObsType:
        return {
            "task": self.task,
            "job": self.job,
            "machine": self.machine,
            "global": self.global_state,
        }

    @override
    def __repr__(self) -> str:
        return (
            f"DefaultObservation("
            f"n_tasks={self.n_tasks}, "
            f"n_jobs={self.n_jobs}, "
            f"n_machines={self.n_machines}"
            f")"
        )

    @override
    def get_spec(self) -> DictSpec:
        return DictSpec(
            {
                "task": DictSpec(
                    {name: self._specs[name] for name in self.task}
                ),
                "job": DictSpec({name: self._specs[name] for name in self.job}),
                "machine": DictSpec(
                    {name: self._specs[name] for name in self.machine}
                ),
                "global": DictSpec(
                    {name: self._specs[name] for name in self.global_state}
                ),
            }
        )

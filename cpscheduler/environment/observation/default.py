"""Default observation for scheduling environments."""

from typing import Any, Literal, TypedDict, overload

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import Status, TaskID, Time
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
    _status: RuntimeFeature[list[int]]
    _available: RuntimeFeature[list[bool]]

    task: dict[str, Any]
    job: dict[str, Any]
    machine: dict[str, Any]
    global_state: dict[str, Any]

    available_tasks: set[TaskID]

    def __init__(
        self,
        exclude_features: set[str] | None = None,
        n_tasks: int | None = None,
        n_machines: int | None = None,
        n_jobs: int | None = None,
        **symbols: int,
    ) -> None:
        """Initialize the DefaultObservation.

        Observations can be initialized with expected symbol values, which can
        be used to have a complete observation spec before any instance has
        been loaded.

        If the inferred symbols do not match the expectations, an error is raised
        during instance loading.

        By default, no symbol has an expected value.

        Parameters
        ----------
        exclude_features: set[str] | None
            A set of feature names to exclude from the observation.
            If None, all features will be included.

        n_tasks: int | None
            Expected number of tasks.

        n_machines: int | None
            Expected number of machines.

        n_jobs: int | None
            Expected number of jobs.
            If n_tasks is specified, but not n_jobs, it is supposed that
            n_jobs = n_tasks.

        **symbols: int
            Additional symbols with expected values.

        """
        super().__init__(n_tasks, n_machines, n_jobs, **symbols)
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

        self._specs = {
            "status": self._status.spec,
            "available": self._available.spec,
            "time": self._time.spec,
        }

        self.task = {
            "status": self._status.value,
            "available": self._available.value,
        }
        self.job = {}
        self.machine = {}
        self.global_state = {"time": 0}

    @property
    def time(self) -> Time:
        """Return the current time in the schedule."""
        return self._time.value

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)
        self.available_tasks = set()

        self._status.value.clear()
        self._available.value.clear()

        self._status.value.extend([AWAITING] * instance.n_tasks)
        self._available.value.extend([False] * instance.n_tasks)

        self.global_state["time"] = self._time.value

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
    def update(self, state: ScheduleState) -> None:
        self._time.value = state.time
        self._status.value[:] = state.runtime.status

        available = self._available.value
        available_tasks = self.available_tasks

        available[:] = [False] * self.n_tasks
        available_tasks.clear()

        for task_id in state.runtime.unlocked_tasks:
            if state.is_available(task_id):
                available[task_id] = True
                available_tasks.add(task_id)

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

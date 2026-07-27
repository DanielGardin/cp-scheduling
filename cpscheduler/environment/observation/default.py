"""Default observation for scheduling environments."""

from collections.abc import Iterable, Mapping
from typing import Any

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import Status, TaskID, Time
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.observation.base import Observation
from cpscheduler.environment.specs import (
    DenseViewSpec,
    DictSpec,
    FeatureViewSpec,
    ObservationSpec,
)
from cpscheduler.environment.state import ScheduleState

DefaultObsType = dict[str, Any]

AWAITING = Status.AWAITING


FEATURE_SELECTION = (
    Iterable[str] | Mapping[str, str | FeatureViewSpec[Any, Any]] | None
)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class DefaultObservation(Observation[DefaultObsType]):
    """Lightweight default observation.

    This is the default observation returned by the environment when none is
    explicitly selected.
    It provides a simple way of accessing the features defined in every component
    of the environment, without the need of defining a custom observation class.

    The features are organized in four scopes: "task", "job", "machine" and "global",
    which can be accessed as dictionaries of feature name to feature value.
    """

    _all_features: bool
    _features: dict[str, str | FeatureViewSpec[Any, Any]]
    feature_specs: dict[str, FeatureViewSpec[Any, Any]]

    _time: Time
    _status: list[int]
    _available: list[bool]

    available_tasks: set[TaskID]

    _obs: DefaultObsType

    def __init__(
        self,
        features: FEATURE_SELECTION = None,
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
        features: Iterable[str] | Mapping[str, str | FeatureViewSpec[Any, Any]] | None
            The selection of features to include in the observation.
            It accepts an iterable of feature names (all default views), or
            a mapping of feature names to either a view name or a custom FeatureViewSpec.
            If None, all features are included with their default views.

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

        self._all_features = features is None
        self._features = {}

        if isinstance(features, Mapping):
            self._features = dict(features)

        elif isinstance(features, Iterable):
            self._features = dict.fromkeys(features, "default")

        self._obs = {}

    @property
    def time(self) -> Time:
        """Return the current time in the schedule."""
        return self._time

    @override
    def compile(self, instance: ProblemInstance) -> ObservationSpec:
        feature_specs: dict[str, FeatureViewSpec[Any, Any]] = {}

        if self._all_features:
            self._features = dict.fromkeys(instance.features.keys(), "default")
            self._features |= dict.fromkeys(
                ["status", "available", "time"], "default"
            )

        if "status" in self._features:
            if self._features["status"] != "default":
                raise ValueError(
                    "The 'status' feature does not have any other view than "
                    "the default."
                )

            feature_specs["status"] = DenseViewSpec(
                value_type="categorical",
                shape=("n_tasks",),
                n_categories=Status.count(),
            )

        if "available" in self._features:
            if self._features["available"] != "default":
                raise ValueError(
                    "The 'available' feature does not have any other view than "
                    "the default."
                )

            feature_specs["available"] = DenseViewSpec(
                value_type="binary",
                shape=("n_tasks",),
            )

        if "time" in self._features:
            if self._features["time"] != "default":
                raise ValueError(
                    "The 'time' feature does not have any other view than "
                    "the default."
                )

            feature_specs["time"] = DenseViewSpec(value_type="time", shape=())

        for feature_name, features in instance.features.items():
            if feature_name not in self._features:
                continue

            view = self._features[feature_name]

            if isinstance(view, FeatureViewSpec):
                feature_specs[feature_name] = view

            else:
                possible_views = features[0].possible_views()

                if view not in possible_views:
                    raise ValueError(
                        f"Feature '{feature_name}' does not have a view named '{view}'."
                    )

                feature_specs[feature_name] = possible_views[view]

        self.feature_specs = feature_specs
        return DictSpec(feature_specs)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)

        self.available_tasks = set()
        self._status = [AWAITING] * instance.n_tasks
        self._available = [False] * instance.n_tasks
        self._time = 0

        obs = self._obs

        obs["status"] = self._status
        obs["available"] = self._available
        obs["time"] = self._time

        for feat_name, features in instance.features.items():
            if feat_name not in self.feature_specs:
                continue

            if not features:
                raise ValueError(
                    f"Feature '{feat_name}' has no values in the instance."
                )

            spec = self.feature_specs[feat_name]
            obs[feat_name] = features[0].materialize(spec, **self.symbols)

    @override
    def update(self, state: ScheduleState) -> None:
        obs = self._obs

        self._time = state.time

        if "time" in obs:
            obs["time"] = state.time

        self._status[:] = state.runtime.status

        available = self._available
        available_tasks = self.available_tasks

        available[:] = [False] * state.n_tasks
        available_tasks.clear()

        for task_id in state.runtime.unlocked_tasks:
            if state.is_available(task_id):
                available[task_id] = True
                available_tasks.add(task_id)

    def __getitem__(self, key: str) -> Any:
        """Get the features of the specified scope."""
        return self._obs[key]

    @override
    def serialize(self) -> DefaultObsType:
        return self._obs

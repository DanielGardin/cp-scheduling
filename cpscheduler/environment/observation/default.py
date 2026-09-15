"""Default observation for scheduling environments."""

from collections.abc import Iterable, Mapping
from typing import Any

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.backend import ScheduleBackend
from cpscheduler.environment.constants import TaskID
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.observation.base import Observation
from cpscheduler.environment.specs import (
    DenseViewSpec,
    DictSpec,
    FeatureViewSpec,
    FreeViewSpec,
    ObservationSpec,
    StackSpec,
)
from cpscheduler.environment.state import ScheduleState

DefaultObsType = dict[str, Any]

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

    _eligible: list[bool]

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
    def eligible_tasks(self) -> list[TaskID]:
        """Return a list with currently eligible tasks."""
        return [
            task_id
            for task_id, eligible in enumerate(self._eligible)
            if eligible
        ]

    @override
    def compile(
        self, instance: ProblemInstance, backend: ScheduleBackend
    ) -> ObservationSpec:
        feature_specs: dict[str, FeatureViewSpec[Any, Any]] = {}

        if self._all_features:
            self._features = dict.fromkeys(
                instance.registered_features.keys(), "default"
            )
            self._features["eligible"] = "default"

        if "eligible" in self._features:
            if self._features["eligible"] != "default":
                raise ValueError(
                    "The 'eligible' feature does not have any other view than "
                    "the default."
                )

            feature_specs["eligible"] = DenseViewSpec(
                value_type="binary",
                shape=("n_tasks",),
            )

        for feature_name, features in instance.registered_features.items():
            if feature_name not in self._features:
                continue

            view = self._features[feature_name]

            if isinstance(view, FeatureViewSpec):
                spec = view

            else:
                possible_views = features[0].possible_views()

                if view not in possible_views:
                    raise ValueError(
                        f"Feature '{feature_name}' does not have a view named '{view}'."
                    )

                spec = possible_views[view]

            feature_specs[feature_name] = spec

        self.feature_specs = feature_specs
        return DictSpec(feature_specs)

    @override
    def initialize(
        self, instance: ProblemInstance, backend: ScheduleBackend
    ) -> None:
        super().initialize(instance, backend)

        self._eligible = [False] * instance.n_tasks

        obs = self._obs

        obs["eligible"] = self._eligible

        for feat_name, features in instance.registered_features.items():
            if feat_name not in self.feature_specs:
                continue

            if not features:
                raise ValueError(
                    f"Feature '{feat_name}' has no values in the instance."
                )

            spec = self.feature_specs[feat_name]
            obs[feat_name] = features[0].materialize(spec, **self.symbols)

    @override
    def update(self, state: ScheduleState, backend: ScheduleBackend) -> None:
        eligible = self._eligible
        eligible[:] = [False] * state.n_tasks

        for task_id in backend.get_eligible_tasks(state):
            eligible[task_id] = True

    def __getitem__(self, key: str) -> Any:
        """Get the features of the specified scope."""
        return self._obs[key]

    @override
    def serialize(self) -> DefaultObsType:
        return self._obs


UNSAFE_SPECS = (FreeViewSpec, StackSpec)


def _select_safe_view(
    views: Mapping[str, FeatureViewSpec[Any, Any]],
) -> str | None:
    """Select a safe view from the given mapping of views.

    This function filters out incompatible views and selects a safe view
    based on the following priority:
    1. Dense views (fixed shape and type)
    2. Views with a fixed shape (but not necessarily dense)
    3. None (if no safe view is found)
    """
    safe_views = [
        (key, spec)
        for key, spec in views.items()
        if not isinstance(spec, UNSAFE_SPECS)
    ]
    if not safe_views:
        return None

    for key, spec in safe_views:
        if isinstance(spec, DenseViewSpec):
            return key

    for key, spec in safe_views:
        if spec.shape is not None:
            return key

    return None


class GymSafeObservation(DefaultObservation):
    """Observation that keeps only Gymnasium-compatible views.

    Some feature views cannot be converted into a Gymnasium `Space`.
    This observation resolves those features to a safe alternative when one is
    available.

    When the features views are given via `features` argument, the observation
    will raise an error if any of the selected views is not compatible with Gymnasium.
    """

    @override
    def compile(
        self, instance: ProblemInstance, backend: ScheduleBackend
    ) -> ObservationSpec:
        feature_specs: dict[str, FeatureViewSpec[Any, Any]] = {}

        if self._all_features:
            self._features = dict.fromkeys(
                instance.registered_features.keys(), "auto"
            )
            self._features["eligible"] = "default"

        if "eligible" in self._features:
            if self._features["eligible"] != "default":
                raise ValueError(
                    "The 'eligible' feature does not have any other view than "
                    "the default."
                )

            feature_specs["eligible"] = DenseViewSpec(
                value_type="binary",
                shape=("n_tasks",),
            )

        for feature_name, features in instance.registered_features.items():
            if feature_name not in self._features:
                continue

            view = self._features[feature_name]

            if isinstance(view, FeatureViewSpec):
                spec = view

            elif view == "auto":
                possible_views = features[0].possible_views()
                safe_view = _select_safe_view(possible_views)
                if safe_view is None:
                    continue

                spec = possible_views[safe_view]

            else:
                possible_views = features[0].possible_views()
                if view not in possible_views:
                    raise ValueError(
                        f"Feature '{feature_name}' does not have a view named '{view}'."
                    )

                spec = possible_views[view]

            if isinstance(spec, UNSAFE_SPECS):
                raise ValueError(
                    f"Feature '{feature_name}''s view {view}"
                    "is not compatible with Gymnasium."
                )

            feature_specs[feature_name] = spec

        self.feature_specs = feature_specs
        return DictSpec(feature_specs)

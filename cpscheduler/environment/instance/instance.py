"""Instance class definition.

Includes the ProblemInstance class, which represents a scheduling problem instance,
and manages its features and data.
The instance provides methods for feature registration, data loading, and validation,
allowing components to interact with the instance's features and data in a structured way.
"""

from typing import TYPE_CHECKING, Any

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    MAX_TIME,
    EzPickle,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.instance.features import (
    Feature,
    TaskFeature,
    merge_symbols,
)
from cpscheduler.environment.specs.feature_spec import FeatureSpec
from cpscheduler.environment.utils.protocols import Instance_T

if TYPE_CHECKING:
    from cpscheduler.environment.setups import ScheduleSetup


def _find_provider(features: list[Feature]) -> Feature | None:
    provider: Feature | None = None
    for feature in features:
        if not feature.owner:
            continue

        if provider is not None:
            raise ValueError(
                f"Feature '{feature.name}' in instance already has a provider."
            )

        provider = feature

    return provider


def _load_data(
    storage: dict[str, Any], features: dict[str, list[Feature]]
) -> dict[str, int]:
    symbol_values: dict[str, int] = {}

    for feature, data in storage.items():
        if feature not in features:
            raise ValueError(
                f"Data provided for feature '{feature}', but no such "
                "feature is registered in the instance."
            )

        feat_list = features[feature]
        for feat in feat_list:
            feat.set_data(data)

        merge_symbols(symbol_values, feat_list[0].solve_symbols())

    return symbol_values


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class ProblemInstance(EzPickle):
    """Class representing a scheduling problem instance.

    The ProblemInstance class manages the features and data of a scheduling
    problem instance.
    It is deeply coupled with the core environment, sharing its state machine:

    State machine
    -------------
    UNLOADED
        During this state, the instance only accepts feature registrations, whose
        specs are unchanged, but data ownership is not yet established.
        This allows components to register features during initialization, without
        needing to know when their features will be loaded with data.

    LOADED
        During this state, the instance accepts data for registered features, and
        shares the loaded data with all consumers.
        The environment transitions to this state when `initialize()` is called, and the
        instance is loaded with data from a new problem instance.

    RUNNING
        While running, the instance is assumed to be frozen.
        Changing instance data during this state may lead to undefined behavior.
        It is not supposed to have the instance as a first-class object during this
        state, instead, the instance is meant to be interacted with via the
        `ScheduleState` class.

    """

    _fingerprint: int
    _feature_specs: dict[str, FeatureSpec]
    features: dict[str, list[Feature]]

    job_tasks: list[list[TaskID]]
    n_tasks: int
    n_jobs: int
    n_machines: int
    symbol_values: dict[str, int]

    _preemptive: TaskFeature[bool]
    _optional: TaskFeature[bool]
    _processing_times: TaskFeature[list[Time]]
    _machine_mask: TaskFeature[list[bool]]
    _job_ids: TaskFeature[TaskID]

    _debug: bool

    def __init__(self, debug_mode: bool) -> None:
        """Initialize an empty ProblemInstance.

        Parameters
        ----------
        debug_mode : bool
            If True, the instance will perform additional checks and validations,
            which may impact performance.

        Notes
        -----
        The instance starts in an unloaded state, where it only accepts feature
        registrations. Data can only be loaded after the instance is initialized with
        an instance using the `initialize()` method, only through the environment's
        loading process, which ensures proper state transitions and data management.

        """
        self.job_tasks = []
        self._preemptive = TaskFeature(
            name="preemptive", semantic="binary", shape=(), owner=True
        )

        self._optional = TaskFeature(
            name="optional", semantic="binary", shape=(), owner=True
        )

        self._machine_mask = TaskFeature(
            name="machine_mask",
            shape=("n_machines",),
            semantic="mask",
            owner=True,
        )

        self._processing_times = TaskFeature(
            name="all_processing_times",
            shape=("n_machines",),
            semantic="duration",
            owner=True,
        )

        self._job_ids = TaskFeature(name="job", semantic="task", shape=())

        # Setting features without self.register(...)
        self.features = {
            "preemptive": [self._preemptive],
            "optional": [self._optional],
            "all_processing_times": [self._processing_times],
            "machine_mask": [self._machine_mask],
            "job": [self._job_ids],
        }

        self._feature_specs = {
            feature_name: features[0].spec
            for feature_name, features in self.features.items()
        }

        self.n_tasks = 0
        self.n_jobs = 0
        self.n_machines = 0
        self.symbol_values = {}
        self._fingerprint = 0

        self._debug = debug_mode

    @property
    def fingerprint(self) -> int:
        """Return the fingerprint of the instance.

        The fingerprint is a unique identifier for the instance, which can be used
        to check if two instances are equivalent.
        """
        return self._fingerprint

    @property
    def debug(self) -> bool:
        """Whether the instance is in debug mode."""
        return self._debug

    @property
    def preemptive(self) -> list[bool]:
        """Preemptivity of each task in the instance.

        Gathered from the `preemptive` feature, has shape (n_tasks,).
        """
        return self._preemptive.value

    @property
    def optional(self) -> list[bool]:
        """Optional status of each task in the instance.

        Gathered from the `optional` feature, has shape (n_tasks,).
        """
        return self._optional.value

    @property
    def machine_mask(self) -> list[list[bool]]:
        """Eligibility mask of machines for each task in the instance.

        Gathered from the `machine_mask` feature, has shape (n_tasks, n_machines).
        Each entry indicates whether the corresponding machine is eligible for the task.
        """
        return self._machine_mask.value

    @property
    def processing_times(self) -> list[list[Time]]:
        """Processing times of each task on each machine in the instance.

        Gathered from the `all_processing_times` feature, has shape (n_tasks, n_machines).
        """
        return self._processing_times.value

    @property
    def job_ids(self) -> list[TaskID]:
        """Job ID of each task in the instance.

        Gathered from the `job` feature, has shape (n_tasks,).
        """
        return self._job_ids.value

    def required_features(self) -> dict[str, FeatureSpec]:
        """Return a dictionary of required features for the instance.

        Required features are those that have no provider among the registered
        features, and thus must be provided with data during instance initialization.
        """
        return {
            name: feature
            for name, feature in self._feature_specs.items()
            if _find_provider(self.features[name]) is None
        }

    def register(self, feature: Feature[Any]) -> None:
        """Register a feature to the instance.

        Registration is a step required after defining a feature inside a component, and
        before loading data into the instance.
        This allows the instance to have static knowledge of all features that
        will be present in the instance, befone any data is provided, and to
        manage data ownership and sharing between features.

        This way, components can define their features, expose them using
        `get_feature()`, and expect data to be available after the instance is
        initialized, without needing to know when or how the data will be loaded,
        as long as the feature is registered before initialization.

        Parameters
        ----------
        feature : Feature[Any]
            The feature to be registered.

        Raises
        ------
        ValueError
            If the feature name is a reserved keyword, or if the feature spec is
            incompatible with a previously registered feature with the same name.

        """
        name = feature.name

        if name in {"all_processing_times", "machine_mask"}:
            raise ValueError(
                f"Name '{feature.name}' is a reserved keyword, if your "
                "component requires access to machine information, retrieve it "
                "directly using `get_processing_time`, and `has_processing_time`."
            )

        registered_spec = self._feature_specs.get(name)

        if registered_spec is None:
            self._feature_specs[name] = feature.spec

        elif registered_spec != feature.spec:
            raise ValueError(f"Incompatible feature spec for '{name}'.")

        self.features.setdefault(name, []).append(feature)

    def unregister(self, feature: Feature[Any]) -> None:
        """Remove a previously registered feature from the instance.

        This is used to clean up instance features that are no longer relevant
        after loading a new instance, allowing features to be re-registered with
        new data without conflicts.
        """
        name = feature.name

        registered = self.features.get(name)
        if not registered:
            return

        registered.remove(feature)

        if not registered:
            del self.features[name]
            del self._feature_specs[name]

    def reset(self) -> None:
        """Reset instance-specific data for loading a new instance.

        Clears loaded state from features and symbol values, allowing
        features to accept new data via set_data() without raising errors.
        Preserves feature registrations and providers.
        """
        self.job_tasks.clear()
        self.n_tasks = 0
        self.n_jobs = 0
        self.n_machines = 0
        self.symbol_values.clear()

        for features in self.features.values():
            for feature in features:
                feature.reset()

        self._fingerprint = 0

    # FUTURE: Memory allocation is currently the major bottleneck during
    # initialization (processing_times and machine_mask).
    # Consider a cached version, which only triggers when the n_tasks, or
    # n_machines in the incoming instance is different from the previous one.
    def initialize(
        self, instances: tuple[Instance_T, ...], setup: "ScheduleSetup"
    ) -> None:
        """Initialize the instance with data from a new problem instance.

        This is the main entry point for loading a new problem instance into the
        environment.

        Parameters
        ----------
        instances : tuple[Instance_T, ...]
            One or more instance data objects to load.

        setup : ScheduleSetup
            The schedule setup, which provides information about the scheduling
            environment, such as the number of machines, and other relevant
            parameters that may be needed during instance initialization.

        """
        self.reset()

        storage: dict[str, Any] = {}

        for instance in instances:
            for feature in instance:
                storage[feature] = instance[feature]

        symbols_values = _load_data(storage, self.features)

        n_tasks = symbols_values.get("n_tasks", 0)
        n_jobs = symbols_values.get("n_jobs", 0)
        n_machines = symbols_values.get("n_machines", setup.n_machines)

        if n_machines != setup.n_machines:
            raise ValueError(
                f"Instance has n_machines={n_machines}, but the setup requires "
                f"n_machines={setup.n_machines}."
            )

        self._preemptive.set_data([False] * n_tasks)
        self._optional.set_data([False] * n_tasks)

        if not self._job_ids.loaded:
            self._job_ids.set_data(list(range(n_tasks)))

        self._processing_times.set_data(
            [[MAX_TIME] * n_machines for _ in range(n_tasks)]
        )
        self._machine_mask.set_data(
            [[False] * n_machines for _ in range(n_tasks)]
        )

        job_ids = self.job_ids

        n_jobs = max(job_ids) + 1 if job_ids else 0
        job_tasks: list[list[TaskID]] = [[] for _ in range(n_jobs)]
        for task_id, job_id in enumerate(job_ids):
            job_tasks[job_id].append(task_id)

        if "n_jobs" in symbols_values:
            if n_jobs != symbols_values["n_jobs"]:
                cur_n_jobs = symbols_values["n_jobs"]

                raise ValueError(
                    f"Instance has 'n_jobs'={cur_n_jobs}, but "
                    f"only {n_jobs} were given by the 'job' feature."
                )

        else:
            symbols_values["n_jobs"] = n_jobs

        self.job_tasks = job_tasks
        self.n_tasks = n_tasks
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.symbol_values = symbols_values

    def finalize(self) -> None:
        """Finalize the instance after loading, where all features are loaded."""
        symbol_values = self.symbol_values

        for name, features in self.features.items():
            provider = _find_provider(features)

            if provider is not None:
                provider_symbols = provider.solve_symbols()
                merge_symbols(symbol_values, provider_symbols)

                for feature in self.features[name]:
                    if feature is provider:
                        continue

                    feature.shared_data(provider)

        self._fingerprint = hash(
            tuple(
                sorted(
                    (name, features[0].compute_hash())
                    for name, features in self.features.items()
                )
            )
        )

    def has_feature(self, feat_name: str) -> bool:
        """Check if a feature with the given name is registered in the instance."""
        return feat_name in self.features

    def get_machines(self, task_id: TaskID) -> list[MachineID]:
        """Get the list of eligible machines for a given task."""
        return [
            m_id
            for m_id, eligible in enumerate(self._machine_mask.value[task_id])
            if eligible
        ]

    def set_processing_time(
        self, task_id: TaskID, machine_id: MachineID, time: Time
    ) -> None:
        """Set the processing time of a task on a specific machine."""
        if time < 0:
            raise ValueError("Processing time cannot be negative.")

        if machine_id >= self.n_machines:
            raise ValueError(
                f"Cannot set processing time for machine {machine_id} in a "
                f"setup with {self.n_machines} machines."
            )

        self._processing_times.value[task_id][machine_id] = time
        self._machine_mask.value[task_id][machine_id] = True

    def remove_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Remove a machine from the eligibility list of a task, making it ineligible for that task."""
        self._machine_mask.value[task_id][machine_id] = False
        self._processing_times.value[task_id][machine_id] = MAX_TIME

    def set_preemption(
        self, task_id: TaskID, allow_preemption: bool = True
    ) -> None:
        """Set the preemptivity of a task in the instance."""
        self._preemptive.value[task_id] = allow_preemption

    def set_optionality(self, task_id: TaskID, optional: bool = True) -> None:
        """Set the optional status of a task in the instance."""
        self._optional.value[task_id] = optional

    def __repr__(self) -> str:
        """Return a string representation of the instance, including its main features and dimensions."""
        features = ", ".join(self.features.keys())

        return (
            f"ProblemInstance("
            f"n_tasks={self.n_tasks}, "
            f"n_jobs={self.n_jobs}, "
            f"n_machines={self.n_machines}, "
            f"features=[{features}])"
        )

    def __eq__(self, value: object, /) -> bool:
        """Check for equality for ProblemInstance objects."""
        return (
            isinstance(value, ProblemInstance)
            and self.features == value.features
            and self._job_ids == value._job_ids
            and self.job_tasks == value.job_tasks
            and self._preemptive == value._preemptive
            and self._optional == value._optional
            and self._processing_times == value._processing_times
            and self.n_tasks == value.n_tasks
            and self.n_jobs == value.n_jobs
            and self.n_machines == value.n_machines
            and self._debug == value._debug
        )

from typing import TYPE_CHECKING, Any, TypeVar

# from mypy_extensions import mypyc_attr
from cpscheduler.environment.constants import EzPickle, MachineID, TaskID, Time
from cpscheduler.environment.instance.features import Feature, TaskFeature
from cpscheduler.environment.utils.protocols import (
    InstanceTypes,
    prepare_instance,
)

if TYPE_CHECKING:
    from cpscheduler.environment.setups import ScheduleSetup

_T = TypeVar("_T")


# TODO: Validate the features after read_instance
class ProblemInstance(EzPickle):
    job_tasks: list[list[TaskID]]
    features: dict[str, list[Feature]]

    _storage: dict[str, Any]
    _providers: dict[str, Feature]
    _symbol_values: dict[str, int]

    _unused_features: dict[str, Any]
    _debug: bool

    _preemptive: TaskFeature[bool]
    _optional: TaskFeature[bool]
    _processing_times: TaskFeature[list[Time]]
    _machine_mask: TaskFeature[list[bool]]
    _job_ids: TaskFeature[TaskID]

    def __init__(self, debug_mode: bool) -> None:
        self._unused_features = {}
        self._storage = {}
        self._symbol_values = {}
        self._debug = debug_mode
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
        self._providers = {
            "preemptive": self._preemptive,
            "optional": self._optional,
            "all_processing_times": self._processing_times,
            "machine_mask": self._machine_mask,
        }

        self.features = {
            "preemptive": [self._preemptive],
            "optional": [self._optional],
            "all_processing_times": [self._processing_times],
            "machine_mask": [self._machine_mask],
            "job": [self._job_ids],
        }

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def n_tasks(self) -> int:
        return self._symbol_values["n_tasks"]

    @property
    def n_jobs(self) -> int:
        return self._symbol_values["n_jobs"]

    @property
    def n_machines(self) -> int:
        return self._symbol_values["n_machines"]

    @property
    def preemptive(self) -> list[bool]:
        return self._preemptive.value

    @property
    def optional(self) -> list[bool]:
        return self._optional.value

    @property
    def machine_mask(self) -> list[list[bool]]:
        return self._machine_mask.value

    @property
    def processing_times(self) -> list[list[Time]]:
        return self._processing_times.value

    @property
    def job_ids(self) -> list[TaskID]:
        return self._job_ids.value

    def register(self, feature: Feature[Any]) -> None:
        name = feature.name

        if name in {"all_processing_times", "machine_mask"}:
            raise ValueError(
                f"Name '{feature.name}' is a reserved keyword, if your "
                "component requires access to machine information, retrieve it "
                "directly using `get_processing_time`, and `has_processing_time`."
            )

        registered = self.features.get(name)

        if registered:
            ref = registered[0]

            if ref.spec != feature.spec:
                raise ValueError(f"Incompatible feature spec for '{name}'.")

        if feature.owner:
            if name in self._providers:
                raise ValueError(
                    "Cannot have two Feature objects with the same name as "
                    f"owners, Feature '{name}' already has a provider."
                )

            self._providers[name] = feature

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

        if self._providers.get(name) is feature:
            del self._providers[name]

        self._unused_features.pop(name, None)

    def validate_instance(self, origin: str) -> None:
        for features in self.features.values():
            for feat in features:
                feat.validate(**self._symbol_values)

    def reset(self) -> None:
        """Reset instance-specific data for loading a new instance.

        Clears loaded state from features and symbol values, allowing
        features to accept new data via set_data() without raising errors.
        Preserves feature registrations and providers.
        """
        self._symbol_values.clear()
        self._storage.clear()
        self._unused_features.clear()
        self.job_tasks.clear()

        # Reset loaded state for all features without defaults
        # (features with defaults can be updated in place)
        for features in self.features.values():
            for feature in features:
                feature.reset()

    def _load_data(self) -> None:
        for name, provider in self._providers.items():
            for feature in self.features[name]:
                if feature is not provider:
                    feature.shared_data(provider)

        for name, data in self._storage.items():
            if name in self._providers:
                raise ValueError(
                    f"Feature '{name}' in instance already has a provider."
                )

            features = self.features.get(name, ())

            for feature in features:
                feature.set_data(data)

            if not features:
                self._unused_features[name] = data

    def initialize(
        self, instance: InstanceTypes, setup: "ScheduleSetup"
    ) -> None:
        if isinstance(instance, tuple):
            task_raw_instance, job_raw_instance = instance

        else:
            task_raw_instance = instance
            job_raw_instance = {}

        task_instance = prepare_instance(task_raw_instance)
        job_instance = prepare_instance(job_raw_instance)

        n_tasks = (
            len(next(iter(task_instance.values()))) if task_instance else 0
        )

        for feat, data in task_instance.items():
            self._storage[feat] = data

        for feat, data in job_instance.items():
            self._storage[feat] = data

        if self._job_ids.name not in self._storage:
            self._job_ids.set_data(list(range(n_tasks)))
            self._job_ids.owner = True
            self._providers[self._job_ids.name] = self._job_ids

        self._preemptive.set_data([False] * n_tasks)
        self._optional.set_data([False] * n_tasks)

        self._load_data()

        # NOTE: These features (machine related) are special, and must not be
        # queried by features, but using the getters of this class.
        # That is because all components have access to these standard
        # information, and using a feature as an alias is a bad practice.
        self._processing_times.set_data(
            [[0] * setup.n_machines for _ in range(n_tasks)]
        )
        self._machine_mask.set_data(
            [[False] * setup.n_machines for _ in range(n_tasks)]
        )

        job_ids = self._job_ids.value

        n_jobs = max(job_ids) + 1 if job_ids else 0
        job_tasks: list[list[TaskID]] = [[] for _ in range(n_jobs)]
        for task_id, job_id in enumerate(job_ids):
            job_tasks[job_id].append(task_id)

        self.job_tasks = job_tasks

        self._symbol_values["n_tasks"] = n_tasks
        self._symbol_values["n_jobs"] = n_jobs
        self._symbol_values["n_machines"] = setup.n_machines

    def has_feature(self, feat_name: str) -> bool:
        return feat_name in self.features or feat_name in self._unused_features

    def get_feature(self, feat_name: str) -> Feature:
        if feat_name not in self.features:
            raise KeyError(
                f"Feature {feat_name} was never registered in the instance."
            )

        return self.features[feat_name][0]

    def get_machines(self, task_id: TaskID) -> list[MachineID]:
        return [
            m_id
            for m_id, eligible in enumerate(self._machine_mask.value[task_id])
            if eligible
        ]

    def set_processing_time(
        self, task_id: TaskID, machine_id: MachineID, time: Time
    ) -> None:
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
        self._machine_mask.value[task_id][machine_id] = False

    def set_preemption(
        self, task_id: TaskID, allow_preemption: bool = True
    ) -> None:
        self._preemptive.value[task_id] = allow_preemption

    def set_optionality(self, task_id: TaskID, optional: bool = True) -> None:
        self._optional.value[task_id] = optional

    def __repr__(self) -> str:
        features = ", ".join(self.features.keys())

        return (
            f"ProblemInstance("
            f"n_tasks={self.n_tasks}, "
            f"n_jobs={self.n_jobs}, "
            f"n_machines={self.n_machines}, "
            f"features=[{features}])"
        )

    def __eq__(self, value: object, /) -> bool:
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

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
        self._providers = {}
        self._symbol_values = {}
        self._debug = debug_mode

        self._preemptive = TaskFeature(name="preemptive", semantic="binary")

        self._optional = TaskFeature(name="optional", semantic="binary")

        self._machine_mask = TaskFeature(
            name="machine_mask",
            shape=("n_machines",),
            semantic="mask",
        )

        self._processing_times = TaskFeature(
            name="all_processing_times",
            shape=("n_machines",),
            semantic="duration",
        )

        self._job_ids = TaskFeature(name="job", semantic="task")

        # Setting features without self.register(...)
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
        return self._symbol_values.get("n_tasks", 0)

    @property
    def n_jobs(self) -> int:
        return self._symbol_values.get("n_jobs", 0)

    @property
    def n_machines(self) -> int:
        return self._symbol_values.get("n_machines", 0)

    @property
    def preemptive(self) -> list[bool]:
        if self._preemptive.loaded:
            return self._preemptive.value

        raise ValueError(
            "ProblemInstance.preemptive: ProblemInstance have not been initialized yet."
        )

    @property
    def optional(self) -> list[bool]:
        if self._optional.loaded:
            return self._optional.value

        raise ValueError(
            "ProblemInstance.optional: ProblemInstance have not been initialized yet."
        )

    @property
    def machine_mask(self) -> list[list[bool]]:
        if self._machine_mask.loaded:
            return self._machine_mask.value

        raise ValueError(
            "ProblemInstance.machine_mask: ProblemInstance have not been initialized yet."
        )

    @property
    def processing_times(self) -> list[list[Time]]:
        if self._processing_times.loaded:
            return self._processing_times.value

        raise ValueError(
            "ProblemInstance.all_processing_times: ProblemInstance have not been initialized yet."
        )

    @property
    def job_ids(self) -> list[TaskID]:
        if self._job_ids.loaded:
            return self._job_ids.value

        raise ValueError(
            "ProblemInstance.all_job_ids: ProblemInstance have not been initialized yet."
        )

    def register(self, feature: Feature[Any]) -> None:
        name = feature.name

        registered = self.features.get(name)

        if registered:
            ref = registered[0]

            if ref.spec != feature.spec:
                raise ValueError(f"Incompatible feature spec for '{name}'.")

        if feature.loaded:
            if name in self._providers:
                raise ValueError(f"Feature '{name}' already has a provider.")

            self._providers[name] = feature

            for consumer in self.features.get(name, ()):
                consumer.shared_data(feature)

        elif name in self._providers:
            # TODO: FeatureSpec guardrails here.
            feature.shared_data(self._providers[name])

        self.features.setdefault(name, []).append(feature)

    def validate_instance(self, origin: str) -> None:
        for features in self.features.values():
            for feat in features:
                feat.validate(**self._symbol_values)

    def _set_instance_data(self, name: str, data: Any) -> None:
        if name in self._providers:
            raise ValueError(
                f"Feature '{name}' in instance already has a provider."
            )

        for feature in self.features.get(name, ()):
            feature.set_data(data)

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
            self._set_instance_data(feat, data)

        for feat, data in job_instance.items():
            self._set_instance_data(feat, data)

        if not self._job_ids.loaded:
            self._job_ids.set_data(list(range(n_tasks)))

        job_ids = self._job_ids.value

        n_jobs = max(job_ids) + 1 if job_ids else 0
        job_tasks: list[list[TaskID]] = [[] for _ in range(n_jobs)]
        for task_id, job_id in enumerate(job_ids):
            job_tasks[job_id].append(task_id)

        self._symbol_values["n_tasks"] = n_tasks
        self._symbol_values["n_jobs"] = n_jobs
        self._symbol_values["n_machines"] = setup.n_machines

        self._preemptive.set_data([False] * n_tasks)
        self._optional.set_data([False] * n_tasks)

        self._processing_times.set_data(
            [[0] * setup.n_machines for _ in range(n_tasks)]
        )
        self._machine_mask.set_data(
            [[False] * setup.n_machines for _ in range(n_tasks)]
        )

        self.job_tasks = job_tasks

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

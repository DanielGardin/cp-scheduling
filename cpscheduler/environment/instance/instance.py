from typing import TYPE_CHECKING, Any, TypeVar

# from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    MachineID, TaskID, Time,
    EzPickle
)

from cpscheduler.environment.instance.features import Feature, TaskFeature

from cpscheduler.environment.utils.general import convert_to_list
from cpscheduler.environment.utils.protocols import (
    InstanceTypes, prepare_instance
)

if TYPE_CHECKING:
    from cpscheduler.environment.schedule_setup import ScheduleSetup

def check_instance_consistency(instance: dict[str, list[Any]]) -> int:
    "Check if all lists in the instance have the same length."
    it = iter(instance.values())
    try:
        first = len(next(it))

    except StopIteration:
        return 0

    for v in it:
        if len(v) != first:
            raise ValueError(
                "Inconsistent instance, all lists must have the same length, "
                f"but got lengths {first} and {len(v)} for keys "
                f"{list(instance.keys())}."
            )

    return first

JOB_FEATURES = ['job', 'job_id']

def get_job_ids(task_instance: dict[str, Any], n_tasks: int) -> list[TaskID]:
    for feat_name in JOB_FEATURES:
        if feat_name in task_instance:
            return convert_to_list(task_instance[feat_name], TaskID)
    
    return convert_to_list(range(n_tasks), TaskID)

_T = TypeVar('_T')

# TODO: Validate the features after read_instance
class ProblemInstance(EzPickle):

    providers: dict[str, Feature]
    features: dict[str, list[Feature]]

    unused_features: dict[str, Any]

    n_tasks: int
    n_jobs: int
    n_machines: int

    # Task definitions
    preemptive: TaskFeature[bool]
    optional: TaskFeature[bool]
    processing_times: TaskFeature[list[Time]]
    machine_mask: TaskFeature[list[bool]]
    job_ids: TaskFeature[TaskID]

    job_tasks: list[list[TaskID]]

    _debug: bool

    def __init__(self, debug_mode: bool) -> None:
        self.unused_features = {}
        self.providers = {}

        self._debug = debug_mode

        self.preemptive = TaskFeature(
            name="preemptive",
            elem_type=bool,
            semantic="binary"
        )

        self.optional = TaskFeature(
            name="optional",
            elem_type=bool,
            semantic="binary"
        )

        self.machine_mask = TaskFeature(
            name="machine_mask",
            elem_type=list[bool],
            semantic="mask",
        )

        self.processing_times = TaskFeature(
            name="all_processing_times",
            elem_type=list[Time],
            semantic="duration",
        )

        self.job_ids = TaskFeature(
            name="job_id",
            elem_type=TaskID,
            semantic="task"
        )

        # Setting features without self.register(...)
        self.features = {
            "preemptive": [self.preemptive],
            "optional": [self.optional],
            "processing_times": [self.processing_times],
            "machine_mask": [self.machine_mask],
            "job_id": [self.job_ids]
        }

    @property
    def debug(self) -> bool:
        return self._debug

    def register(self, feature: Feature[Any]) -> None:
        name = feature.name

        registered = self.features.get(name)

        if registered:
            ref = registered[0]

            if ref.spec != feature.spec:
                raise ValueError(
                    f"Incompatible feature spec for '{name}'."
                )

        if feature.loaded:
            if name in self.providers:
                raise ValueError(
                    f"Feature '{name}' already has a provider."
                )

            self.providers[name] = feature

            for consumer in self.features.get(name, ()):
                consumer.shared_data(feature)

        elif name in self.providers:
            # TODO: FeatureSpec guardrails here.
            feature.shared_data(self.providers[name])

        self.features.setdefault(name, []).append(feature)

    def _validate_instance(self, origin: str) -> None:
        for name, features in self.features.items():
            for feat in features:
                if feat.spec.optional or feat.loaded:
                    continue
            
                raise ValueError(
                    f"{origin}: Feature '{name}' is mandatory, but was not loaded. "
                    "Check your instance specification, or your components."
                )

    def _set_instance_data(self, name: str, data: Any) -> None:
        if name in self.providers:
            raise ValueError(
                f"Feature '{name}' in instance already has a provider."
            )

        for feature in self.features.get(name, ()):
            feature.set_data(data)

    def read_instance(self, instance: InstanceTypes, setup: "ScheduleSetup") -> None:
        if isinstance(instance, tuple):
            task_raw_instance, job_raw_instance = instance

        else:
            task_raw_instance = instance
            job_raw_instance = {}


        task_instance = prepare_instance(task_raw_instance)
        job_instance = prepare_instance(job_raw_instance)

        n_tasks = check_instance_consistency(task_instance)
        check_instance_consistency(job_instance)

        for feat, data in task_instance.items():
            self._set_instance_data(feat, data)

        for feat, data in job_instance.items():
            self._set_instance_data(feat, data)

        job_ids = get_job_ids(task_instance, n_tasks)

        n_jobs = max(job_ids) + 1 if job_ids else 0
        job_tasks: list[list[TaskID]] = [[] for _ in range(n_jobs)]
        for task_id, job_id in enumerate(job_ids):
            job_tasks[job_id].append(task_id)

        self.n_tasks = n_tasks
        self.n_jobs = n_jobs
        self.n_machines = setup.n_machines

        self.preemptive.set_data([False] * n_tasks)
        self.optional.set_data([False] * n_tasks)

        self.processing_times.set_data([
            [0] * setup.n_machines for _ in range(n_tasks)
        ])
        self.machine_mask.set_data([
            [False] * setup.n_machines for _ in range(n_tasks)
        ])

        self.job_ids.set_data(job_ids)
        self.job_tasks = job_tasks

        self._validate_instance("read_instance")

    def has_feature(self, feat_name: str) -> bool:
        return (
            feat_name in self.features
            or feat_name in self.unused_features
        )

    def get_feature(self, feat_name: str) -> Feature:
        if feat_name not in self.features:
            raise KeyError(
                f"Feature {feat_name} was never registered in the instance."
            )

        return self.features[feat_name][0]

    def get_machines(self, task_id: TaskID) -> list[MachineID]:
        return [
            m_id
            for m_id, eligible in enumerate(self.machine_mask.value[task_id])
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

        self.processing_times.value[task_id][machine_id] = time
        self.machine_mask.value[task_id][machine_id] = True

    def remove_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        self.machine_mask.value[task_id][machine_id] = False

    def set_preemption(self, task_id: TaskID, allow_preemption: bool = True) -> None:
        self.preemptive.value[task_id] = allow_preemption

    def set_optionality(self, task_id: TaskID, optional: bool = True) -> None:
        self.optional.value[task_id] = optional

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, ProblemInstance)
            and self.features == value.features
            and self.job_ids == value.job_ids
            and self.job_tasks == value.job_tasks
            and self.preemptive == value.preemptive
            and self.optional == value.optional
            and self.processing_times == value.processing_times
            and self.n_tasks == value.n_tasks
            and self.n_jobs == value.n_jobs
            and self.n_machines == value.n_machines
            and self._debug == value._debug
        )

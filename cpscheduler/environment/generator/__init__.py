from __future__ import annotations

"""Registry-based instance generation for scheduling environments.

The generator inspects the environment setup, constraints, passive constraints,
and objective, then resolves builder specs from explicit registries. This keeps
instance creation extensible without relying on protocol-style feature checks.
"""

from collections.abc import Callable, Iterable, Mapping
from random import Random
from typing import Any, ClassVar, cast

from cpscheduler.environment._protocols import InstanceTypes
from cpscheduler.environment.constraints import (
    DeadlineConstraint,
    NonRenewableResourceConstraint,
    OptionalityConstraint,
    PreemptionConstraint,
    ReleaseDateConstraint,
    ResourceConstraint,
)
from cpscheduler.environment.objectives import (
    MaximumLateness,
    TotalEarliness,
    TotalFlowTime,
    TotalTardiness,
    TotalTardyJobs,
    WeightedCompletionTime,
    WeightedEarliness,
    WeightedTardiness,
    WeightedTardyJobs,
)
from cpscheduler.environment.schedule_setup import (
    FlexibleJobShopSetup,
    FlowShopSetup,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    OpenShopSetup,
    ScheduleSetup,
    SingleMachineSetup,
    UniformParallelMachineSetup,
    UnrelatedParallelMachineSetup,
)

SetupBuilder = Callable[..., dict[str, list[Any]]]
ComponentBuilder = Callable[..., None]
SetupBuilderSelection = str | SetupBuilder | tuple[str | SetupBuilder, Mapping[str, Any]]
ComponentBuilderSelection = (
    str | ComponentBuilder | tuple[str | ComponentBuilder, Mapping[str, Any]]
)


__all__ = [
    "EnvSpecInstanceGenerator",
    "ReleaseDateUniformBuilder",
    "ReleaseDateFromProcessingBuilder",
    "DeadlineConstantBuilder",
    "DeadlineFromReleaseAndProcessingBuilder",
]


def _shuffle(rng: Random, items: list[int]) -> list[int]:
    copied = items.copy()
    rng.shuffle(copied)
    return copied


class ReleaseDateUniformBuilder:
    def __init__(self, low: int = 0, high: int = 15) -> None:
        if high < low:
            raise ValueError("high must be >= low.")

        self.low = int(low)
        self.high = int(high)

    def __call__(
        self,
        component: Any,
        instance: dict[str, list[Any]],
        env: Any,
        rng: Random,
        generator: "EnvSpecInstanceGenerator",
        **params: Any,
    ) -> None:
        low = int(params.get("low", self.low))
        high = int(params.get("high", self.high))
        if high < low:
            raise ValueError("high must be >= low.")

        n_tasks = generator._task_count(instance)
        instance.setdefault(
            component.release_tag,
            [rng.randint(low, high) for _ in range(n_tasks)],
        )


class ReleaseDateFromProcessingBuilder:
    def __init__(
        self,
        processing_tag: str = "processing_time",
        min_slack: int = 5,
        max_slack: int = 40,
    ) -> None:
        if max_slack < min_slack:
            raise ValueError("max_slack must be >= min_slack.")

        self.processing_tag = processing_tag
        self.min_slack = int(min_slack)
        self.max_slack = int(max_slack)

    def __call__(
        self,
        component: Any,
        instance: dict[str, list[Any]],
        env: Any,
        rng: Random,
        generator: "EnvSpecInstanceGenerator",
        **params: Any,
    ) -> None:
        processing_tag = cast(str, params.get("processing_tag", self.processing_tag))
        min_slack = int(params.get("min_slack", self.min_slack))
        max_slack = int(params.get("max_slack", self.max_slack))
        if max_slack < min_slack:
            raise ValueError("max_slack must be >= min_slack.")

        n_tasks = generator._task_count(instance)
        processing = instance.get(processing_tag, [0] * n_tasks)

        instance.setdefault(
            component.release_tag,
            [
                int(proc) + rng.randint(min_slack, max_slack)
                for proc in processing
            ],
        )


class DeadlineConstantBuilder:
    def __init__(self, due_date: int) -> None:
        self.due_date = int(due_date)

    def __call__(
        self,
        component: Any,
        instance: dict[str, list[Any]],
        env: Any,
        rng: Random,
        generator: "EnvSpecInstanceGenerator",
        **params: Any,
    ) -> None:
        due_date = int(params.get("due_date", self.due_date))
        n_tasks = generator._task_count(instance)
        instance.setdefault(
            component.due_tag,
            [due_date for _ in range(n_tasks)],
        )


class DeadlineFromReleaseAndProcessingBuilder:
    def __init__(
        self,
        processing_tag: str = "processing_time",
        release_tag: str = "release_time",
        min_slack: int = 5,
        max_slack: int = 40,
    ) -> None:
        if max_slack < min_slack:
            raise ValueError("max_slack must be >= min_slack.")

        self.processing_tag = processing_tag
        self.release_tag = release_tag
        self.min_slack = int(min_slack)
        self.max_slack = int(max_slack)

    def __call__(
        self,
        component: Any,
        instance: dict[str, list[Any]],
        env: Any,
        rng: Random,
        generator: "EnvSpecInstanceGenerator",
        **params: Any,
    ) -> None:
        processing_tag = cast(str, params.get("processing_tag", self.processing_tag))
        release_tag = cast(str, params.get("release_tag", self.release_tag))
        min_slack = int(params.get("min_slack", self.min_slack))
        max_slack = int(params.get("max_slack", self.max_slack))
        if max_slack < min_slack:
            raise ValueError("max_slack must be >= min_slack.")

        n_tasks = generator._task_count(instance)
        processing = instance.get(processing_tag, [0] * n_tasks)
        release = instance.get(release_tag, [0] * n_tasks)

        instance.setdefault(
            component.due_tag,
            [
                int(r) + int(p) + rng.randint(min_slack, max_slack)
                for r, p in zip(release, processing)
            ],
        )


def _build_optional_flags(
    component: Any,
    instance: dict[str, list[Any]],
    env: Any,
    rng: Random,
    generator: "EnvSpecInstanceGenerator",
) -> None:
    if component.all_tasks:
        return

    if component.optionality_tag:
        n_tasks = generator._task_count(instance)
        instance.setdefault(
            component.optionality_tag,
            [rng.random() < generator.optional_probability for _ in range(n_tasks)],
        )


def _build_preemption_flags(
    component: Any,
    instance: dict[str, list[Any]],
    env: Any,
    rng: Random,
    generator: "EnvSpecInstanceGenerator",
) -> None:
    if component.all_tasks:
        return

    if component.preemption_tag:
        n_tasks = generator._task_count(instance)
        instance.setdefault(
            component.preemption_tag,
            [rng.random() < generator.preemption_probability for _ in range(n_tasks)],
        )


def _build_weight_feature(
    component: Any,
    instance: dict[str, list[Any]],
    env: Any,
    rng: Random,
    generator: "EnvSpecInstanceGenerator",
    *,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
) -> None:
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight.")

    n_tasks = generator._task_count(instance)
    instance.setdefault(
        component.weight_tag,
        [rng.uniform(min_weight, max_weight) for _ in range(n_tasks)],
    )


def _build_weights_feature(
    component: Any,
    instance: dict[str, list[Any]],
    env: Any,
    rng: Random,
    generator: "EnvSpecInstanceGenerator",
    *,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
) -> None:
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight.")

    n_tasks = generator._task_count(instance)
    instance.setdefault(
        component.weights_tag,
        [rng.uniform(min_weight, max_weight) for _ in range(n_tasks)],
    )


def _build_release_feature(
    component: Any,
    instance: dict[str, list[Any]],
    env: Any,
    rng: Random,
    generator: "EnvSpecInstanceGenerator",
    *,
    min_release_time: int = 0,
    max_release_time: int = 15,
) -> None:
    if max_release_time < min_release_time:
        raise ValueError("max_release_time must be >= min_release_time.")

    n_tasks = generator._task_count(instance)
    instance.setdefault(
        component.release_tag,
        [
            rng.randint(min_release_time, max_release_time)
            for _ in range(n_tasks)
        ],
    )


def _build_due_feature(
    component: Any,
    instance: dict[str, list[Any]],
    env: Any,
    rng: Random,
    generator: "EnvSpecInstanceGenerator",
    *,
    min_slack: int = 5,
    max_slack: int = 40,
) -> None:
    if max_slack < min_slack:
        raise ValueError("max_slack must be >= min_slack.")

    n_tasks = generator._task_count(instance)
    processing_tag = generator._infer_processing_tag(instance)
    processing = instance[processing_tag] if processing_tag is not None else [1] * n_tasks

    if component.due_tag in instance:
        return

    instance[component.due_tag] = [
        int(proc) + rng.randint(min_slack, max_slack)
        for proc in processing
    ]


def _build_resource_usage(
    component: Any,
    instance: dict[str, list[Any]],
    env: Any,
    rng: Random,
    generator: "EnvSpecInstanceGenerator",
    *,
    resource_usage_ratio: float = 0.6,
) -> None:
    n_tasks = generator._task_count(instance)

    for tag, capacity in zip(component.resource_tags, component.capacities):
        if tag in instance:
            continue

        cap = max(0.0, float(capacity) * resource_usage_ratio)
        instance[tag] = [rng.uniform(0.0, cap) for _ in range(n_tasks)]


class EnvSpecInstanceGenerator:
    n_jobs: int

    optional_probability: float
    preemption_probability: float

    default_n_machines: int | None
    resource_usage_ratio: float

    setup_builders: ClassVar[dict[type[Any], dict[str, SetupBuilder]]] = {}
    setup_default_builders: ClassVar[dict[type[Any], str]] = {}
    component_builders: ClassVar[dict[type[Any], dict[str, ComponentBuilder]]] = {}
    component_default_builders: ClassVar[dict[type[Any], str]] = {}

    def __init__(
        self,
        n_jobs: int,
        *,
        optional_probability: float = 0.2,
        preemption_probability: float = 0.2,
        default_n_machines: int | None = None,
        resource_usage_ratio: float = 0.6,
        setup_builder: SetupBuilderSelection | None = None,
        selected_builders: Mapping[type[Any], ComponentBuilderSelection] | None = None,
    ) -> None:
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1.")

        if not 0.0 <= optional_probability <= 1.0:
            raise ValueError("optional_probability must be in [0, 1].")

        if not 0.0 <= preemption_probability <= 1.0:
            raise ValueError("preemption_probability must be in [0, 1].")

        if resource_usage_ratio < 0.0:
            raise ValueError("resource_usage_ratio must be >= 0.")

        self.n_jobs = n_jobs
        self.optional_probability = optional_probability
        self.preemption_probability = preemption_probability
        self.default_n_machines = default_n_machines
        self.resource_usage_ratio = resource_usage_ratio

        self.setup_builder = setup_builder
        self.selected_builders = dict(selected_builders or {})

    @classmethod
    def register_setup_builder(
        cls,
        setup_type: type[Any],
        builder: SetupBuilder | None = None,
        *,
        name: str | None = None,
        default: bool = False,
    ) -> Any:
        if builder is None:
            def decorator(fn: SetupBuilder) -> SetupBuilder:
                cls.register_setup_builder(setup_type, fn, name=name, default=default)
                return fn

            return decorator

        builder_name = cast(str, name or getattr(builder, "__name__", builder.__class__.__name__))
        cls.setup_builders.setdefault(setup_type, {})[builder_name] = builder

        if default or setup_type not in cls.setup_default_builders:
            cls.setup_default_builders[setup_type] = builder_name

        return builder

    @classmethod
    def register_component_builder(
        cls,
        component_type: type[Any],
        builder: ComponentBuilder | None = None,
        *,
        name: str | None = None,
        default: bool = False,
    ) -> Any:
        if builder is None:
            def decorator(fn: ComponentBuilder) -> ComponentBuilder:
                cls.register_component_builder(
                    component_type, fn, name=name, default=default
                )
                return fn

            return decorator

        builder_name = cast(str, name or getattr(builder, "__name__", builder.__class__.__name__))
        cls.component_builders.setdefault(component_type, {})[builder_name] = builder

        if default or component_type not in cls.component_default_builders:
            cls.component_default_builders[component_type] = builder_name

        return builder

    def select_builder(
        self,
        component_type: type[Any],
        builder: ComponentBuilderSelection,
    ) -> None:
        self.selected_builders[component_type] = builder

    @staticmethod
    def _split_builder_selection(
        selection: str | Callable[..., Any] | tuple[str | Callable[..., Any], Mapping[str, Any]],
    ) -> tuple[str | Callable[..., Any], dict[str, Any]]:
        if isinstance(selection, tuple):
            builder_spec, params = selection
            return builder_spec, dict(params)

        return selection, {}

    @staticmethod
    def _find_registry_entry(
        component: Any,
        registry: Mapping[type[Any], dict[str, Callable[..., Any]]],
    ) -> tuple[type[Any], dict[str, Callable[..., Any]]] | None:
        for cls in type(component).__mro__:
            if cls in registry:
                return cls, registry[cls]

        return None

    def _resolve_setup_builder(
        self,
        setup: ScheduleSetup,
    ) -> tuple[SetupBuilder, dict[str, Any]] | None:
        if self.setup_builder is not None:
            builder_spec, builder_params = self._split_builder_selection(self.setup_builder)
            if isinstance(builder_spec, str):
                entry = self._find_registry_entry(setup, self.setup_builders)
                if entry is None:
                    return None

                _, builders = entry
                resolved = cast(SetupBuilder | None, builders.get(builder_spec))
                if resolved is None:
                    return None

                return resolved, builder_params

            return cast(SetupBuilder, builder_spec), builder_params

        entry = self._find_registry_entry(setup, self.setup_builders)
        if entry is None:
            return None

        component_type, builders = entry
        default_name = self.setup_default_builders.get(component_type)
        if default_name is None:
            return None

        resolved = cast(SetupBuilder | None, builders.get(default_name))
        if resolved is None:
            return None

        return resolved, {}

    def _resolve_component_builder(
        self,
        component: Any,
        selection: Mapping[type[Any], ComponentBuilderSelection],
        registry: Mapping[type[Any], dict[str, ComponentBuilder]],
        defaults: Mapping[type[Any], str],
    ) -> tuple[ComponentBuilder, dict[str, Any]] | None:
        for cls in type(component).__mro__:
            spec = selection.get(cls)
            if spec is not None:
                builder_spec, builder_params = self._split_builder_selection(spec)
                if isinstance(builder_spec, str):
                    builders = registry.get(cls)
                    if builders is None:
                        continue

                    resolved = builders.get(builder_spec)
                    if resolved is None:
                        continue

                    return resolved, builder_params

                return cast(ComponentBuilder, builder_spec), builder_params

        for cls in type(component).__mro__:
            default_name = defaults.get(cls)
            if default_name is None:
                continue

            builders = registry.get(cls)
            if builders is None:
                continue

            resolved = builders.get(default_name)
            if resolved is None:
                continue

            return resolved, {}

        return None

    def _resolve_selected_component_builder(
        self, component: Any
    ) -> tuple[ComponentBuilder, dict[str, Any]] | None:
        return self._resolve_component_builder(
            component,
            self.selected_builders,
            self.component_builders,
            self.component_default_builders,
        )

    def _resolve_registered_component_builder(
        self, component: Any
    ) -> tuple[ComponentBuilder, dict[str, Any]] | None:
        return self._resolve_component_builder(
            component,
            {},
            self.component_builders,
            self.component_default_builders,
        )

    def _machine_count(self, setup: ScheduleSetup, env: Any) -> int:
        if env.state.loaded and env.state.n_machines > 0:
            return int(env.state.n_machines)

        if setup.n_machines > 0:
            return int(setup.n_machines)

        if self.default_n_machines is not None and self.default_n_machines > 0:
            return self.default_n_machines

        raise ValueError(
            "Could not infer number of machines from setup. "
            "Provide default_n_machines in EnvSpecInstanceGenerator."
        )

    def _rand_int(
        self,
        rng: Random,
        *,
        min_processing_time: int = 1,
        max_processing_time: int = 20,
    ) -> int:
        if min_processing_time < 0 or max_processing_time < min_processing_time:
            raise ValueError("Invalid processing-time range.")

        return rng.randint(min_processing_time, max_processing_time)

    def _task_count(self, data: Mapping[str, list[Any]]) -> int:
        if data:
            return len(next(iter(data.values())))

        return self.n_jobs

    @staticmethod
    def _infer_processing_tag(data: Mapping[str, list[Any]]) -> str | None:
        for key in data:
            if key.startswith("processing_time") or key == "processing_time":
                return key

        if data:
            return next(iter(data))

        return None

    def _build_parallel_instance(
        self,
        setup: Any,
        rng: Random,
        **params: Any,
    ) -> dict[str, list[Any]]:
        n_tasks = self.n_jobs

        if isinstance(setup, UnrelatedParallelMachineSetup):
            return {
                tag: [self._rand_int(rng, **params) for _ in range(n_tasks)]
                for tag in setup.processing_times
            }

        return {
            setup.processing_times: [self._rand_int(rng, **params) for _ in range(n_tasks)],
        }

    def _build_jobshop_like_instance(
        self,
        setup: Any,
        env: Any,
        rng: Random,
        **params: Any,
    ) -> dict[str, list[Any]]:
        n_machines = self._machine_count(setup, env)
        n_tasks = self.n_jobs * n_machines

        jobs = [j for j in range(self.n_jobs) for _ in range(n_machines)]
        operations = [o for _ in range(self.n_jobs) for o in range(n_machines)]

        data: dict[str, list[Any]] = {
            "job": jobs,
            setup.operation_order: operations,
        }

        if isinstance(setup, FlexibleJobShopSetup):
            for p_tag in setup.processing_times:
                data[p_tag] = [self._rand_int(rng, **params) for _ in range(n_tasks)]
        else:
            data[setup.processing_times] = [self._rand_int(rng, **params) for _ in range(n_tasks)]

        if isinstance(setup, FlowShopSetup):
            machines = operations
        else:
            machines = []
            for _ in range(self.n_jobs):
                machines.extend(_shuffle(rng, list(range(n_machines))))

        data[setup.machine_feature] = machines
        return data

    def _build_openshop_instance(
        self,
        setup: OpenShopSetup,
        env: Any,
        rng: Random,
        **params: Any,
    ) -> dict[str, list[Any]]:
        n_machines = self._machine_count(setup, env)
        n_tasks = self.n_jobs * n_machines

        jobs = [j for j in range(self.n_jobs) for _ in range(n_machines)]
        machines: list[int] = []
        for _ in range(self.n_jobs):
            machines.extend(_shuffle(rng, list(range(n_machines))))

        return {
            "job": jobs,
            setup.machine_feature: machines,
            setup.processing_times: [self._rand_int(rng, **params) for _ in range(n_tasks)],
        }

    def _setup_instance(self, env: Any, rng: Random) -> dict[str, list[Any]]:
        setup = env.setup

        resolved = self._resolve_setup_builder(setup)
        if resolved is not None:
            builder, builder_params = resolved
            return builder(setup, env, rng, self, **builder_params)

        if isinstance(
            setup,
            (
                SingleMachineSetup,
                IdenticalParallelMachineSetup,
                UniformParallelMachineSetup,
                UnrelatedParallelMachineSetup,
            ),
        ):
            return self._build_parallel_instance(setup, rng)

        if isinstance(setup, (JobShopSetup, FlowShopSetup, FlexibleJobShopSetup)):
            return self._build_jobshop_like_instance(setup, env, rng)

        if isinstance(setup, OpenShopSetup):
            return self._build_openshop_instance(setup, env, rng)

        raise NotImplementedError(
            f"No random generator rule implemented for setup '{type(setup).__name__}'."
        )

    def _components_to_build(self, env: Any) -> Iterable[Any]:
        return (*env.constraints, *env.setup_constraints, *env.passive_constraints, env.objective)

    def _apply_component_builders(
        self,
        data: dict[str, list[Any]],
        components: Iterable[Any],
        env: Any,
        rng: Random,
    ) -> None:
        for component in components:
            selected = self._resolve_selected_component_builder(component)
            if selected is not None:
                selected_builder, selected_params = selected
                selected_builder(component, data, env, rng, self, **selected_params)

            default = self._resolve_registered_component_builder(component)
            if default is not None:
                default_builder, default_params = default
                if selected is None or default_builder is not selected[0]:
                    default_builder(component, data, env, rng, self, **default_params)

    def _register_builtin_builders(self) -> None:
        if self.component_builders:
            return

        self.register_component_builder(
            ReleaseDateConstraint,
            ReleaseDateUniformBuilder(),
            name="uniform",
            default=True,
        )
        self.register_component_builder(
            ReleaseDateConstraint,
            ReleaseDateFromProcessingBuilder(),
            name="from_processing",
        )

        self.register_component_builder(
            DeadlineConstraint,
            DeadlineFromReleaseAndProcessingBuilder(),
            name="from_release_and_processing",
            default=True,
        )

        due_only_types: tuple[type[Any], ...] = (
            TotalEarliness,
            TotalTardiness,
            TotalTardyJobs,
            MaximumLateness,
        )
        for component_type in due_only_types:
            self.register_component_builder(component_type, _build_due_feature, default=True)

        due_and_weight_types: tuple[type[Any], ...] = (
            WeightedEarliness,
            WeightedTardiness,
            WeightedTardyJobs,
        )
        for component_type in due_and_weight_types:
            self.register_component_builder(component_type, _build_due_feature, default=True)

        for component_type in (
            WeightedCompletionTime,
        ):
            self.register_component_builder(component_type, _build_weights_feature, default=True)

        for component_type in due_and_weight_types:
            self.register_component_builder(component_type, _build_weight_feature, default=True)

        self.register_component_builder(TotalFlowTime, _build_release_feature, default=True)

        self.register_component_builder(OptionalityConstraint, _build_optional_flags, default=True)
        self.register_component_builder(PreemptionConstraint, _build_preemption_flags, default=True)
        self.register_component_builder(ResourceConstraint, _build_resource_usage, default=True)
        self.register_component_builder(NonRenewableResourceConstraint, _build_resource_usage, default=True)

    def sample(self, env: Any, *, seed: int | None = None) -> InstanceTypes:
        self._register_builtin_builders()

        rng = Random(seed)
        data = self._setup_instance(env, rng)
        components = tuple(self._components_to_build(env))

        self._apply_component_builders(data, components, env, rng)

        return data

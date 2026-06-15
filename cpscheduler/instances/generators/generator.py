"""Generic feature-based instance generator."""

from random import Random
from typing import Any, cast

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.specs import FeatureSpec
from cpscheduler.environment.specs.symbols import resolve_shape, symbolic_shape
from cpscheduler.environment.utils.protocols import InstanceGenerator
from cpscheduler.instances.distributions.base import Sampler
from cpscheduler.instances.generators.default import DEFAULT_SAMPLERS


class FeatureSampler(EzPickle):
    """Helper class to associate a sampler with a feature and its spec."""

    name: str
    sampler: Sampler[Any]
    outer_shape: tuple[int, ...]

    def __init__(
        self,
        name: str,
        shape: tuple[int | None, ...] | None,
        sampler: Sampler[Any],
        **symbols: int,
    ) -> None:
        """Initialize a FeatureSampler.

        Parameters
        ----------
        name: str
            The name of the feature being sampled.

        shape: tuple[int | None, ...] | None
            The expected shape of the feature value.
            None is treated as a wildcard, and any sample is accepted.

        sampler: Sampler[Any]
            The sampler to generate values for this feature.
            The sampler's shape must be compatible with the provided shape.

        **symbols: int
            Additional symbols that may be needed to resolve the sampler's shape.

        Raises
        ------
        ValueError
            If the sampler's shape is not compatible with the provided shape.

        """
        self.name = name
        self.sampler = sampler

        sample_shape = resolve_shape(symbolic_shape(sampler.shape), **symbols)

        self.outer_shape = _computer_outer_shape(sample_shape, shape)

    @classmethod
    def from_spec(cls, name: str, spec: FeatureSpec, sampler: Sampler[Any], **symbols: int) -> "FeatureSampler":
        """Create a FeatureSampler from a feature spec and a sampler."""
        return cls(
            name=name,
            shape=spec.resolve_shape(**symbols),
            sampler=sampler,
            **symbols
        )

    def sample(self, rng: Random, **context: Any) -> Any:
        """Sample a value for this feature using the associated sampler.

        The resulting shape of the sampled value is obtained by potentially
        adding outer dimensions to match the expected shape, resulting in many
        values being i.i.d. samples from the provided sampler.
        """
        return self._sample_shape(rng, context,)

    def _sample_shape(
        self,
        rng: Random,
        context: dict[str, Any],
        depth: int = 0,
    ) -> Any:
        if depth == len(self.outer_shape):
            return self.sampler.sample(
                rng=rng,
                **context,
            )

        length = self.outer_shape[depth]
        return [
            self._sample_shape(
                rng,
                context,
                depth + 1,
            )
            for _ in range(length)
        ]


_Shape = tuple[int | None, ...] | None

def _computer_outer_shape(shape: _Shape, target_shape: _Shape) -> tuple[int, ...]:
    if target_shape is None:
        return ()

    if shape is None:
        if not target_shape or target_shape[-1] is not None:
            raise ValueError(
                f"Shape {shape} is not compatible with target shape {target_shape}: "
                f"shape is None but target shape has last dimension {target_shape[-1]}."
            )

        outer = target_shape[:-1]

        if any(dim is None for dim in outer):
            raise ValueError(
                f"Target shape {target_shape} has None in outer dimensions, "
                "loop shape cannot be determined."
            )

        return cast("tuple[int, ...]", outer)


    if len(shape) > len(target_shape):
        raise ValueError(
            f"Shape {shape} is not compatible with target shape {target_shape}: "
            "shape has more dimensions than target shape."
        )

    for s_dim, t_dim in zip(reversed(shape), reversed(target_shape), strict=False):
        if t_dim is not None and s_dim != t_dim:
            raise ValueError(
                f"Shape {shape} is not compatible with target shape {target_shape}: "
                f"dimension {s_dim} does not match target dimension {t_dim}."
            )

    n_outer = len(target_shape) - len(shape)
    outer = target_shape[:n_outer]

    if any(dim is None for dim in outer):
        raise ValueError(
            f"Target shape {target_shape} has None in outer dimensions, "
            "loop shape cannot be determined."
        )

    return cast("tuple[int, ...]", outer)

class Generator(EzPickle, InstanceGenerator):
    """Generic feature-based instance generator.

    The generator itself contains no scheduling assumptions, every required
    feature must have a registered sampler.

    Samplers receive the current generation context and may depend on
    previously generated features.
    """

    feature_specs: dict[str, FeatureSpec]
    _sampling_order: tuple[str, ...]

    _symbols: dict[str, int]
    _samplers: dict[str, FeatureSampler]
    _rng: Random

    def __init__(
        self,
        feature_specs: dict[str, FeatureSpec],
        n_tasks: int,
        n_machines: int,
        n_jobs: int | None = None,
        *,
        samplers: dict[str, Sampler[Any]] | None = None,
        seed: int | None = None,
        use_default_samplers: bool = True,
        **symbols: int,
    ) -> None:
        """Initialize a Generator.

        Parameters
        ----------
        feature_specs: dict[str, FeatureSpec]
            The feature specifications defining the required features for the
            environment. Each feature must have a registered sampler.

        n_tasks: int
            The number of tasks in the generated instances. This is a required
            symbol that can be used in feature specs and samplers.

        n_machines: int
            The number of machines in the generated instances. This is a required
            symbol that can be used in feature specs and samplers.

        n_jobs: int | None, optional
            The number of jobs in the generated instances. If `None`, it is
            assumed to be equal to `n_tasks`. This is a symbol that can be used in
            feature specs and samplers.

        samplers: dict[str, Sampler[Any]] | None, optional
            A dictionary mapping feature names to samplers. If `None`, no samplers
            are registered initially, and all features must be registered manually
            using the `register` method.

        seed: int | None, optional
            An optional random seed for reproducibility. If `None`, the generator
            will be initialized without a fixed seed.

        use_default_samplers: bool, optional
            Whether to use default samplers for features when available. If `True`,
            the generator will attempt to register default samplers for any features
            that do not have a sampler provided in the `samplers` argument.

        **symbols: int
            Additional symbols that can be used in feature specs and samplers.
            Currently unused.

        """
        n_jobs = n_jobs if n_jobs is not None else n_tasks

        self.feature_specs = feature_specs

        if n_jobs == n_tasks:
            # If n_jobs is not explicitly specified, it is assumed to be equal
            # to n_tasks. In this case, we remove 'job' from the feature specs,
            # since it is deterministically defined as the same as the task index.
            self.feature_specs.pop("job", None)

        elif n_tasks < n_jobs:
            raise ValueError(
                f"Number of tasks (n_tasks={n_tasks}) cannot be less than number "
                f"of jobs (n_jobs={n_jobs})."
            )

        _symbols = {
            "n_tasks": n_tasks,
            "n_machines": n_machines,
            "n_jobs": n_jobs,
            **symbols,
        }

        _default_samplers = self._default_samplers() if use_default_samplers else {}

        _samplers: dict[str, FeatureSampler] = {}

        for name, feature_spec in self.feature_specs.items():
            if samplers and name in samplers:
                _samplers[name] = FeatureSampler.from_spec(
                    name=name,
                    spec=feature_spec,
                    sampler=samplers[name],
                    **_symbols,
                )

            elif name in _default_samplers:
                _samplers[name] = FeatureSampler.from_spec(
                    name=name,
                    spec=feature_spec,
                    sampler=_default_samplers[name],
                    **_symbols,
                )

        self._sampling_order = ()
        self._symbols = _symbols
        self._samplers = _samplers
        self._rng = Random(seed)

    @classmethod
    def from_env(
        cls,
        env: SchedulingEnv,
        n_tasks: int,
        n_machines: int | None = None,
        n_jobs: int | None = None,
        *,
        samplers: dict[str, Sampler[Any]] | None = None,
        seed: int | None = None,
        use_default_samplers: bool = True,
        **symbols: int,
    ) -> "Generator":
        """Create a Generator from a scheduling environment.

        Parameters
        ----------
        env: SchedulingEnv
            The scheduling environment for which to create the generator. The
            environment's required features will be used as the generator's
            feature specifications.

        n_tasks: int
            The number of tasks in the generated instances.

        n_machines: int | None, optional
            The number of machines in the generated instances. If `None`, it is
            inferred from the environment.

        n_jobs: int | None, optional
            The number of jobs in the generated instances. If `None`, it is
            assumed to be equal to `n_tasks`.

        samplers: dict[str, Sampler[Any]] | None, optional
            A dictionary mapping feature names to samplers. If `None`, no samplers
            are registered initially, and all features must be registered manually
            using the `register` method.

        seed: int | None, optional
            An optional random seed for reproducibility. If `None`, the generator
            will be initialized without a fixed seed.

        use_default_samplers: bool, optional
            Whether to use default samplers for features when available. If `True`,
            the generator will attempt to register default samplers for any features
            that do not have a sampler provided in the `samplers` argument.

        **symbols: int
            Additional symbols that can be used in feature specs and samplers.
            Currently unused.

        """
        if n_machines is None:
            n_machines = env.setup.n_machines

            if n_machines == 0:
                raise ValueError(
                    "Number of machines must be specified. Scheduling setup "
                    f"{type(env.setup).__name__} does not specify a fixed "
                    f"number of machines."
                )

        return cls(
            feature_specs=env.required_features(),
            n_tasks=n_tasks,
            n_machines=n_machines,
            n_jobs=n_jobs,
            samplers=samplers,
            seed=seed,
            use_default_samplers=use_default_samplers,
            **symbols,
        )

    def _default_samplers(self) -> dict[str, Sampler[Any]]:
        return DEFAULT_SAMPLERS

    def _invalidate_ordering(self) -> None:
        self._sampling_order = ()

    @property
    def sampling_order(self) -> tuple[str, ...]:
        """Valid sampling order of features based on their dependencies."""
        if self._sampling_order:
            return self._sampling_order

        in_degree: dict[str, int] = dict.fromkeys(self.feature_specs, 0)
        dependencies: dict[str, list[str]] = {name: [] for name in self.feature_specs}

        for name in self.feature_specs:
            if name not in self._samplers:
                raise ValueError(
                    f"No sampler provided for feature '{name}'. A sampler must be "
                    "registered for every required feature."
                )

            sampler = self._samplers[name].sampler

            for dep in sampler.dependencies:
                if dep not in self.feature_specs:
                    raise ValueError(
                        f"Sampler for feature '{name}' depends on feature '{dep}', "
                        "which is not in the environment's required features."
                    )

                in_degree[name] += 1
                dependencies[dep].append(name)

        order: list[str] = []
        zero_in_degree = [name for name, degree in in_degree.items() if degree == 0]

        while zero_in_degree:
            name = zero_in_degree.pop()
            order.append(name)

            for dep in dependencies[name]:
                degree = in_degree[dep] - 1
                in_degree[dep] = degree

                if degree == 0:
                    zero_in_degree.append(dep)

        if len(order) != len(self.feature_specs):
            missing = set(self.feature_specs) - set(order)
            raise ValueError(
                f"Circular dependencies detected among features: {missing}."
            )

        sampling_order = tuple(order)

        self._sampling_order = sampling_order
        return sampling_order

    def register(
        self,
        feature_name: str,
        sampler: Sampler[Any],
    ) -> None:
        """Register a sampler for a feature.

        Parameters
        ----------
        feature_name: str
            The name of the feature to register the sampler for.

        sampler: Sampler[Any]
            The sampler to register for the feature.

        """
        self._samplers[feature_name] = FeatureSampler.from_spec(
            name=feature_name,
            spec=self.feature_specs[feature_name],
            sampler=sampler,
            **self._symbols,
        )
        self._invalidate_ordering()

    def unregister(
        self,
        feature_name: str,
    ) -> None:
        """Unregister the sampler for a feature."""
        self._samplers.pop(feature_name, None)
        self._invalidate_ordering()

    def sample(
        self,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Sample a new scheduling instance with the registered features.

        Parameters
        ----------
        seed: int | None, optional
            An optional random seed for reproducibility. If `None`, the generator's
            internal random state is used without modification.

        Returns
        -------
        dict[str, Any]
            A dictionary mapping feature names to their sampled values, representing
            a complete scheduling instance according to the environment's required
            features.

        """
        if seed is not None:
            self._rng.seed(seed)

        instance: dict[str, Any] = {}

        context: dict[str, Any] = dict(self._symbols)

        for feature_name in self.sampling_order:
            spec = self.feature_specs[feature_name]
            sampler = self._samplers[feature_name]

            context["feature_name"] = feature_name
            context["spec"] = spec

            feature_value = sampler.sample(self._rng, **context)

            instance[feature_name] = feature_value
            context[feature_name] = feature_value

        return instance

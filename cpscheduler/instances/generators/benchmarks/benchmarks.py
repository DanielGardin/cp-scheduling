"""Benchmark registry class for benchmarking instance configuration."""

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, NoReturn, final

from cpscheduler.common import AnySchedulingEnv
from cpscheduler.instances.distributions.base import Sampler
from cpscheduler.instances.generators.generator import Generator

SamplerFactory = Callable[..., Mapping[str, Sampler[Any]]]


@final
class Benchmark:
    """Registry of named sampler-set factories."""

    _registry: ClassVar[dict[str, SamplerFactory]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[SamplerFactory], SamplerFactory]:
        r"""Register a new benchmark instance generator sampler.

        Usage:
        >>> Benchmark.register("my_benchmark"):
        >>> def _my_benchmark_fn(low: int, high: int):
        >>>     mean_p = (high + low) / 2
        >>>
        >>>     return {
        >>>         "processing_time": UniformInt(low, high),
        >>>         "due_time": UniformInt(
        >>>             f"{int(0.7*mean_p)}*n_tasks",
        >>>             f"{int(1.3*mean_p)}*n_tasks",
        >>>         )
        >>>     }

        Then, you can generate instance by
        >>> gen = Benchmark.create(env, "my_benchmark", n_tasks=100)
        >>> gen.sample()
        """

        def decorator(fn: SamplerFactory) -> SamplerFactory:
            if name in cls._registry:
                raise ValueError(f"Benchmark '{name}' is already registered.")

            cls._registry[name] = fn
            return fn

        return decorator

    def __new__(cls) -> NoReturn:
        """Benchmark is a register class, do not instantiate it."""
        raise ValueError(
            "Cannot instantiate Benchmark, use `Benchmark,create instead`"
        )

    @classmethod
    def create(
        cls,
        env: AnySchedulingEnv,
        name: str,
        *,
        n_tasks: int | None = None,
        n_machines: int | None = None,
        n_jobs: int | None = None,
        seed: int | None = None,
        use_default_samplers: bool = True,
        symbols: Mapping[str, int] | None = None,
        **sampler_kwargs: Any,
    ) -> Generator:
        """Create a Generator with the Benchmark configuration.

        Parameters
        ----------
        env: AnySchedulingEnv
            The environment to generate a instance to.

        name: str
            Name of the benchmark. Must be registered first.

        n_tasks: int | None, optional
            The number of tasks in the generated instances.
            If `None`, it is inferred from the environment.

        n_machines: int | None, optional
            The number of machines in the generated instances. If `None`, it is
            inferred from the environment.

        n_jobs: int | None, optional
            The number of jobs in the generated instances. If `None`, it is
            assumed to be equal to `n_tasks`.

        seed: int | None, optional
            An optional random seed for reproducibility. If `None`, the generator
            will be initialized without a fixed seed.

        use_default_samplers: bool, optional
            Whether to use default samplers for features. If `True`, the generator
            will use default samplers, inferred from the feature metadata.
            If `False`, it will raise an error if a sampler is not provided for
            a feature.

        symbols: Mapping[str, int], optional
            Additional symbols that can be used in feature specs and samplers.

        **sampler_kwargs: Any
            The parameters used in the benchmark configuration.

        Returns
        -------
        generator: Generator
            The benchmark instance generator

        """
        if name not in cls._registry:
            raise KeyError(
                f"Unknown benchmark '{name}'. Available: {sorted(cls._registry)}"
            )

        samplers = cls._registry[name](**sampler_kwargs)

        return Generator.from_env(
            env,
            n_tasks=n_tasks,
            n_machines=n_machines,
            n_jobs=n_jobs,
            samplers=samplers,
            seed=seed,
            use_default_samplers=use_default_samplers,
            **(symbols or {}),
        )

    @classmethod
    def apply(
        cls,
        env: AnySchedulingEnv,
        name: str,
        *,
        seed: int | None = None,
        n_tasks: int | None = None,
        n_machines: int | None = None,
        n_jobs: int | None = None,
        use_default_samplers: bool = True,
        symbols: Mapping[str, int] | None = None,
        **sampler_kwargs: Any,
    ) -> Generator:
        """Create and set a Generator with the Benchmark configuration to the env.

        Parameters
        ----------
        env: AnySchedulingEnv
            The environment to generate a instance to.

        name: str
            Name of the benchmark. Must be registered first.

        n_tasks: int | None, optional
            The number of tasks in the generated instances.
            If `None`, it is inferred from the environment.

        n_machines: int | None, optional
            The number of machines in the generated instances. If `None`, it is
            inferred from the environment.

        n_jobs: int | None, optional
            The number of jobs in the generated instances. If `None`, it is
            assumed to be equal to `n_tasks`.

        seed: int | None, optional
            An optional random seed for reproducibility. If `None`, the generator
            will be initialized without a fixed seed.

        use_default_samplers: bool, optional
            Whether to use default samplers for features. If `True`, the generator
            will use default samplers, inferred from the feature metadata.
            If `False`, it will raise an error if a sampler is not provided for
            a feature.

        symbols: Mapping[str, int], optional
            Additional symbols that can be used in feature specs and samplers.

        **sampler_kwargs: Any
            The parameters used in the benchmark configuration.

        Returns
        -------
        generator: Generator
            The benchmark instance generator

        """
        generator = cls.create(
            env,
            name,
            seed=seed,
            n_tasks=n_tasks,
            n_machines=n_machines,
            n_jobs=n_jobs,
            use_default_samplers=use_default_samplers,
            symbols=symbols,
            **sampler_kwargs,
        )

        env.set_generator(generator)
        return generator

    @classmethod
    def available(cls) -> list[str]:
        """List of currently available benchmarks."""
        return sorted(cls._registry)

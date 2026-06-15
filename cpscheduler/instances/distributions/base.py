"""
Base abstractions for stochastic sampling in scheduling instance generation.

This module defines the foundational interfaces used throughout the instance
generation subsystem. The design intentionally separates:
- scalar probability distributions,
- structured stochastic processes,

All randomness is externally controlled through explicit RNG injection,
ensuring reproducibility and composability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable, MutableSequence, Sequence
    from random import Random

    from cpscheduler.environment.specs.symbols import BaseShapeDim


_T_co = TypeVar("_T_co", covariant=True)
_U = TypeVar("_U")


class Sampler(ABC, Generic[_T_co]):
    """Base stochastic sampling interface.

    A Sampler is any object capable of generating values using a provided
    random state and optional contextual information.

    Examples
    --------
    - scalar distributions,
    - stochastic processes,
    - graph generators,
    - machine eligibility generators,
    - routing generators.

    Samplers must be stateless with respect to randomness. All randomness
    must originate from the provided RNG object.
    """

    @property
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        """The resulting shape after sampling.

        Implies the shape of the object returned by `sample`.
        When the shape is undecidable, return None (default).
        """
        return None

    @property
    def dependencies(self) -> tuple[str, ...]:
        """List of feature names that this sampler depends on.

        This is used for dependency resolution during generation, where
        the sampler requires certain features to be generated beforehand and
        passed in the context.

        By default, samplers have no dependencies.
        """
        return ()

    @abstractmethod
    def sample(self, rng: Random, *args: Any, **context: Any) -> _T_co:
        """Generate a sample.

        Parameters
        ----------
        rng: Random
            Random number generator used for reproducibility.

        *args: Any
            Ignored during sampling.

        **context: Any
            Optional contextual parameters used during generation.
            Reserved keys:
            - "n_tasks"
            - "n_jobs"
            - "n_machines"
            - "feature_name"
            - "spec"
            - dependency feature names (described in the `dependencies` property)

        """
        raise NotImplementedError

    def __call__(self, rng: Random, *args: Any, **context: Any) -> _T_co:
        """Callable shorthand for sample()."""
        return self.sample(rng, *args, **context)

    def map(
        self,
        transform: Callable[[_T_co], _U],
    ) -> Sampler[_U]:
        """Return a new sampler that applies a transformation to this sampler's output."""
        return Mapped(self, transform)

    def filter(
        self,
        predicate: Callable[[_T_co], bool],
        max_attempts: int = 1000,
    ) -> Sampler[_T_co]:
        """Return a new sampler that filters this sampler's output using a predicate."""
        return RejectionSampler(self, predicate, max_attempts)

    def __repr__(self) -> str:
        """Return a string representation of the sampler."""
        return f"{self.__class__.__name__}()"


class Mapped(Sampler[_T_co]):
    """
    Apply a transformation to samples.

    Example
    -------
    >>> duration = Uniform(0.0, 100.0)
    >>> rounded = duration.map(round)
    """

    sampler: Sampler[Any]
    transform: Callable[[Any], _T_co]
    _shape: tuple[BaseShapeDim, ...] | None

    def __init__(
        self,
        sampler: Sampler[_U],
        transform: Callable[[_U], _T_co],
        final_shape: tuple[BaseShapeDim, ...] | None = None,
    ) -> None:
        """Initialize the Mapped sampler.

        Parameters
        ----------
        sampler: Sampler[_U]
            The underlying sampler to generate base samples from.

        transform: Callable[[_U], _T_co]
            A function that transforms samples from the base sampler to the desired output type.

        final_shape: tuple[BaseShapeDim, ...] | None, optional
            The shape of the output after transformation.
            If None, the shape is erased and defaults to None.

        """
        self.sampler = sampler
        self.transform = transform
        self._shape = final_shape

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return self._shape

    @property
    @override
    def dependencies(self) -> tuple[str, ...]:
        return self.sampler.dependencies

    @override
    def sample(
        self,
        rng: Random,
        **context: Any,
    ) -> _T_co:
        return self.transform(self.sampler.sample(rng, **context))

    @override
    def __repr__(self) -> str:
        transform_name = getattr(
            self.transform,
            "__name__",
            self.transform.__class__.__name__,
        )

        return f"Mapped({self.sampler!r}, {transform_name})"


class RejectionSampler(Sampler[_T_co]):
    """
    Rejection sampling wrapper.

    Example
    -------
    >>> positive = RejectionSampler(
    ...     Normal(0, 10),
    ...     lambda x: x > 0,
    ... )
    """

    sampler: Sampler[_T_co]
    predicate: Callable[[_T_co], bool]
    max_attempts: int

    def __init__(
        self,
        sampler: Sampler[_T_co],
        predicate: Callable[[_T_co], bool],
        max_attempts: int = 1000,
    ) -> None:
        """Initialize the RejectionSampler.

        Parameters
        ----------
        sampler: Sampler[_T_co]
            The underlying sampler to generate candidate samples from.

        predicate: Callable[[_T_co], bool]
            A function that takes a sample and returns True if it is accepted, False otherwise.

        max_attempts: int, default=1000
            The maximum number of attempts to generate a valid sample before raising an error.

        """
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self.sampler = sampler
        self.predicate = predicate
        self.max_attempts = max_attempts

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return self.sampler.shape

    @property
    @override
    def dependencies(self) -> tuple[str, ...]:
        return self.sampler.dependencies

    def sample(
        self,
        rng: Random,
        **context: Any,
    ) -> _T_co:
        """Generate a sample.

        Parameters
        ----------
        rng: Random
            Random number generator used for reproducibility.

        *args: Any
            Ignored during sampling.

        **context: Any
            Optional contextual parameters used during generation.
            Reserved keys:
            - "n_tasks"
            - "n_jobs"
            - "n_machines"
            - "feature_name"
            - "spec"
            - dependency feature names (described in the `dependencies` property)

        Raises
        ------
        RuntimeError
            If a valid sample cannot be generated within the maximum number of attempts.

        """
        for _ in range(self.max_attempts):
            value = self.sampler.sample(
                rng,
                **context,
            )

            if self.predicate(value):
                return value

        raise RuntimeError(
            "Failed to generate a valid sample after "
            f"{self.max_attempts} attempts."
        )

    @override
    def __repr__(self) -> str:
        return f"Filtered({self.sampler!r})"


# Distributions
# ------------------------------------------------------------------------------

NumericType = int | float | bool

_N_co = TypeVar("_N_co", bound=NumericType)
_N = TypeVar("_N", bound=NumericType)
_M = TypeVar("_M", bound=NumericType)
_O = TypeVar("_O", bound=NumericType)


class Distribution(Sampler[_N_co], ABC):
    """Base class for scalar numeric probability distributions.

    Distributions are intended for iid value generation:
    - processing times,
    - setup costs,
    - weights,
    - priorities,
    - release gaps,
    - due date factors.

    The output is always interpreted as a single scalar.
    They should not encode structural scheduling logic.
    """

    @final
    @property
    def shape(self) -> tuple[()]:
        """The resulting shape after sampling.

        Distributions always produce scalar outputs, so the shape is always ().
        """
        return ()

    @final
    @property
    def dependencies(self) -> tuple[()]:
        """List of feature names that this distribution depends on.

        Distributions are intended to be independent of other features, so
        they must have no dependencies.
        """
        return ()

    def map_numeric(self, transform: Callable[[_N_co], _M]) -> Distribution[_M]:
        """Return a new distribution that applies a numeric transformation to this distribution's output."""
        return MappedDistribution(self, transform)


class MappedDistribution(Distribution[_O]):
    """Apply a transformation to a distribution.

    Example
    -------
    >>> duration = Uniform(0.0, 100.0)
    >>> rounded = MappedDistribution(duration, round)
    """

    distribution: Distribution[Any]
    transform: Callable[[Any], _O]

    def __init__(
        self,
        distribution: Distribution[_N],
        transform: Callable[[_N], _O],
    ) -> None:
        """Initialize the MappedDistribution.

        Parameters
        ----------
        distribution: Distribution[_N]
            The underlying distribution to generate base samples from.

        transform: Callable[[_N], _O]
            A function that transforms samples from the base distribution to the desired output type.

        """
        self.distribution = distribution
        self.transform = transform

    @override
    def sample(
        self,
        rng: Random,
        **context: Any,
    ) -> _O:
        return self.transform(self.distribution.sample(rng, **context))

    @override
    def __repr__(self) -> str:
        transform_name = getattr(
            self.transform,
            "__name__",
            self.transform.__class__.__name__,
        )

        return f"MappedDistribution({self.distribution!r}, {transform_name})"


# Processes
# ------------------------------------------------------------------------------

_MS = TypeVar("_MS", bound="MutableSequence[Any]")


class Process(Sampler[_T_co], ABC):
    """Base class for structured stochastic processes.

    Processes generate correlated or structured outputs where samples
    depend on broader scheduling context.

    Examples
    --------
    - Poisson arrival streams,
    - precedence graph generation,
    - machine eligibility topology,
    - routing structures,
    - bottleneck patterns,
    - correlated job families.

    Unlike scalar distributions, processes often generate collections,
    graphs, schedules, or feature tensors directly.
    """

    def shuffle(self: Process[_MS]) -> Process[_MS]:
        """Return a new process that shuffles the output of this process.

        This method is only valid for processes that output mutable sequences.
        """
        return Shuffled(self)


class Shuffled(Process[_MS]):
    """
    Shuffle the output of another process.

    Example
    -------
    >>> release_times = PoissonProcess(10)
    >>> shuffled_release_times = Shuffled(release_times)
    """

    process: Process[_MS]

    def __init__(self, process: Process[_MS]) -> None:
        """Initialize the Shuffled process.

        Parameters
        ----------
        process: Process[_MS]
            The underlying process to generate base samples from.
            It must output a mutable sequence that can be shuffled in place.

        """
        self.process = process

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return self.process.shape

    @property
    @override
    def dependencies(self) -> tuple[str, ...]:
        return self.process.dependencies

    @override
    def sample(
        self,
        rng: Random,
        **context: Any,
    ) -> _MS:
        values = self.process.sample(rng, **context)
        rng.shuffle(values)

        return values

    @override
    def __repr__(self) -> str:
        return f"Shuffled({self.process!r})"


_T = TypeVar("_T")


class JobPartitionProcess(Process[list[_T]]):
    """Samples from a process identically and independently for each job.

    The inner process is expected to generate a sequence of values for a single
    job, receiving as context:
    - n_tasks: number of tasks in the job,
    - n_jobs: 1,


    Example
    -------
    >>> release_times = PoissonProcess(10)
    >>> job_processing_times = JobPartitionProcess(release_times)
    """

    process: Process[Sequence[_T]]
    contiguous_jobs: bool
    shuffle_tasks: bool

    def __init__(
        self,
        process: Process[Sequence[_T]],
        contiguous_jobs: bool = True,
        shuffle_tasks: bool = False,
    ) -> None:
        """Initialize the JobPartitionProcess.

        Parameters
        ----------
        process: Process[Sequence[_T]]
            The underlying process to sample from for each job.
            It must have no outer dependencies other than symbolic context parameters.

        contiguous_jobs: bool, default=True
            If True, assumes that tasks belonging to the same job are contiguous
            in the output. This allows for more efficient sampling and is suitable
            for most scheduling scenarios where tasks of the same job are grouped
            together.

        shuffle_tasks: bool, default=False
            If True, shuffles tasks within each job after sampling.

        """
        if process.dependencies:
            raise ValueError(
                "The inner process must have no dependencies. "
                f"Found dependencies: {process.dependencies}"
            )

        self.process = process
        self.contiguous_jobs = contiguous_jobs
        self.shuffle_tasks = shuffle_tasks

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return ("n_tasks",)

    @property
    @override
    def dependencies(self) -> tuple[str, ...]:
        return ("job",)

    @override
    def sample(
        self,
        rng: Random,
        *,
        n_tasks: int,
        n_jobs: int,
        job: Sequence[int],
        **context: Any,
    ) -> list[_T]:
        if len(job) != n_tasks:
            raise ValueError(
                f"Length of context 'job' must be equal to n_tasks. "
                f"Got {len(job)} and {n_tasks}."
            )

        counts = [0] * n_jobs
        for job_id in job:
            if not (0 <= job_id < n_jobs):
                raise ValueError(
                    f"Invalid job ID {job_id} in context 'job'. "
                    f"Expected values in [0, {n_jobs - 1}]."
                )

            counts[job_id] += 1

        # Output buffer to hold samples in job order:
        # [job0_task0, job0_task1, ..., job1_task0, job1_task1, ...]
        output: list[_T] = []
        for count in counts:
            if count > 0:
                job_context = context.copy()
                job_context["n_tasks"] = count
                job_context["n_jobs"] = 1
                job_context["job"] = [0] * count

                job_values = self.process.sample(
                    rng,
                    **job_context,
                )

                if self.shuffle_tasks:
                    job_values = list(job_values)
                    rng.shuffle(job_values)

                output.extend(job_values)

        if self.contiguous_jobs:
            return output

        # Handle the case where jobs are not contiguous
        # Job cursor tracks the next insertion index for each job, i.e.
        # [job0_task0, job0_task1, ..., job1_task0, job1_task1, ...]
        #  ^ job_cursor[0],             ^ job_cursor[1]         ...]
        job_cursor = [0] * n_jobs
        for i in range(n_jobs - 1):
            job_cursor[i + 1] = job_cursor[i] + counts[i]

        result: list[_T] = [None] * n_tasks  # type: ignore[list-item]
        for task_id, job_id in enumerate(job):
            result[task_id] = output[job_cursor[job_id]]
            job_cursor[job_id] += 1

        return result

    def shuffle_within_jobs(self) -> JobPartitionProcess[_T]:
        """Return a new JobPartitionProcess that shuffles tasks within each job."""
        return JobPartitionProcess(
            process=self.process,
            contiguous_jobs=self.contiguous_jobs,
            shuffle_tasks=True,
        )

"""Structured samplers used in scheduling instances."""

from math import fabs, floor, lgamma, log, log2, sqrt
from random import Random
from typing import Any

from typing_extensions import override

from cpscheduler.environment.utils.symbols import BaseShapeDim
from cpscheduler.instances.distributions.base import Process
from cpscheduler.instances.distributions.discrete import Multinomial


# This implementation is a copy of the `random.Random.binomialvariate` method,
# which is not available in Python <=3.11.
def _binomialvariate(rng: Random, n: int = 1, p: float = 0.5) -> int:
    # Error check inputs and handle edge cases
    if n < 0:
        raise ValueError("n must be non-negative")
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return 0
        if p == 1.0:
            return n
        raise ValueError("p must be in the range 0.0 <= p <= 1.0")

    # Fast path for a common case
    if n == 1:
        return int(rng.random() < p)

    # Exploit symmetry to establish:  p <= 0.5
    if p > 0.5:
        return n - _binomialvariate(rng, n, 1.0 - p)

    if n * p < 10.0:
        # BG: Geometric method by Devroye with running time of O(np).
        # https://dl.acm.org/doi/pdf/10.1145/42372.42381
        x = y = 0
        c = log2(1.0 - p)
        if not c:
            return x

        while True:
            y += floor(log2(rng.random()) / c) + 1
            if y > n:
                return x

            x += 1

    # BTRS: Transformed rejection with squeeze method by Wolfgang Hörmann
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8407&rep=rep1&type=pdf
    assert n * p >= 10.0
    assert p <= 0.5

    spq = sqrt(n * p * (1.0 - p))  # Standard deviation of the distribution
    b = 1.15 + 2.53 * spq
    a = -0.0873 + 0.0248 * b + 0.01 * p
    c = n * p + 0.5
    vr = 0.92 - 4.2 / b

    setup_complete = False

    alpha = 0.0
    lpq = 0.0
    m = 0
    h = 0.0
    while True:
        u = rng.random()
        u -= 0.5
        us = 0.5 - fabs(u)
        k = floor((2.0 * a / us + b) * u + c)
        if k < 0 or k > n:
            continue

        # The early-out "squeeze" test substantially reduces
        # the number of acceptance condition evaluations.
        v = rng.random()
        if us >= 0.07 and v <= vr:
            return k

        # Acceptance-rejection test.
        # Note, the original paper erroneously omits the call to log(v)
        # when comparing to the log of the rescaled binomial distribution.
        if not setup_complete:
            alpha = (2.83 + 5.1 / b) * spq
            lpq = log(p / (1.0 - p))
            m = floor((n + 1) * p)  # Mode of the distribution
            h = lgamma(m + 1) + lgamma(n - m + 1)
            setup_complete = True  # Only needs to be done once

        v *= alpha / (a / (us * us) + b)
        if log(v) <= h - lgamma(k + 1) - lgamma(n - k + 1) + (k - m) * lpq:
            return k


class PoissonProcess(Process[list[int]]):
    """Discrete Poisson arrival process for job/task release times.

    Generates event times using exponential inter-arrival gaps.

    Arguments
    ---------
    rate: float
        The number of tasks released, on average, at each time unit.

    loc: int | None
        The time of the first arrival.
        If None, the first arrival follows the Exponential distribution.

    Context
    -------
    n_tasks: int

    """

    rate: float
    loc: int | None

    def __init__(self, rate: float, loc: int | None = None) -> None:
        """Initialize a PoissonProcess.

        Parameters
        ----------
        rate: float
            The number of tasks released, on average, at each time unit.

        loc: int | None
            The time of the first arrival.
            If None, the first arrival follows the Exponential distribution.

        Raises
        ------
        ValueError
            If `rate` is not > 0.

        """
        if rate <= 0:
            raise ValueError("rate must be > 0")

        self.rate = rate
        self.loc = loc

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return ("n_tasks",)

    @override
    def sample(self, rng: Random, *, n_tasks: int, **context: Any) -> list[int]:
        t: float = (
            float(self.loc)
            if self.loc is not None
            else rng.expovariate(self.rate)
        )

        arrivals: list[int] = [int(t)]

        for _ in range(1, n_tasks):
            gap = rng.expovariate(self.rate)
            t += gap
            arrivals.append(int(t))

        return arrivals

    @override
    def __repr__(self) -> str:
        return f"PoissonProcess(rate={self.rate}, loc={self.loc})"


class UniformMachineEligibility(Process[list[list[bool]]]):
    """Generate a uniform machine eligibility selection.

    The output is a dictionary of tasks, with a list of all the eligible
    machines for that task.

    Context
    -------
    n_machines: int
    n_tasks: int

    """

    p: float

    def __init__(self, p: float) -> None:
        """Initialize a UniformMachineEligibility process.

        Parameters
        ----------
        p: float
            The probability that a given machine is eligible for a given task.

        Raises
        ------
        ValueError
            If `p` is not in the range [0, 1].

        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], received {p}.")

        self.p = p

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return ("n_tasks", "n_machines")

    @override
    def sample(
        self, rng: Random, *, n_tasks: int, n_machines: int, **context: Any
    ) -> list[list[bool]]:
        return [
            [rng.random() < self.p for _ in range(n_machines)]
            for _ in range(n_tasks)
        ]

    @override
    def __repr__(self) -> str:
        return f"MachineEligibilityMask(p={self.p})"


class BernoulliPrecedence(Process[dict[int, list[int]]]):
    """Generate a random directed acyclic graph (DAG) of precedence constraints.

    This uses a simple approach of sampling DAGs by selecting an edge (i, j),
    with i < j, with a fixed probability p.

    Note that this process assumes that tasks are indexed in a topological order,

    Context
    -------
    n_tasks: int

    """

    p: float

    def __init__(self, p: float) -> None:
        """Initialize a BernoulliPrecedence process.

        Parameters
        ----------
        p: float
            The probability that a precedence constraint exists between any pair of tasks.

        Raises
        ------
        ValueError
            If `p` is not in the range [0, 1].

        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], received {p}.")

        self.p = p

    @override
    def sample(
        self, rng: Random, *, n_tasks: int, **context: Any
    ) -> dict[int, list[int]]:
        precedence: dict[int, list[int]] = {}

        for child in range(1, n_tasks):
            n_parents = _binomialvariate(rng, child, self.p)

            if n_parents:
                precedence[child] = rng.sample(range(child), n_parents)

        return precedence

    @override
    def __repr__(self) -> str:
        return f"BernoulliPrecedence(p={self.p})"


class JobAssignmentProcess(Process[list[int]]):
    """Generate a random job assignment for tasks.

    This process generates a random assignment of tasks to jobs, where each
    task is assigned to a job uniformly at random from the set of available
    jobs.

    Context
    -------
    n_tasks: int
    n_jobs: int

    """

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return ("n_tasks",)

    @override
    def sample(
        self, rng: Random, *, n_tasks: int, n_jobs: int, **context: Any
    ) -> list[int]:
        multinomial_sampler = Multinomial(
            n_tasks - n_jobs, [1 / n_jobs] * n_jobs
        )

        job_counts = multinomial_sampler.sample(rng)

        job_assignment: list[int] = [-1] * n_tasks
        task_id = 0
        for job_id, count in enumerate(job_counts):
            for _ in range(count + 1):
                job_assignment[task_id] = job_id
                task_id += 1

        return job_assignment

    @override
    def __repr__(self) -> str:
        return "JobAssignmentProcess()"

"""Deterministic processes for scheduling problems."""

from random import Random
from typing import Any

from typing_extensions import override

from cpscheduler.environment.specs.symbols import BaseShapeDim
from cpscheduler.instances.distributions.base import Process


class DeterministicJobAssignment(Process[list[int]]):
    """
    Deterministic job assignment process.

    Assigns each task to a specific machine in a packed round-robin fashion.
    The output is a contiguous assignment of tasks to jobs.

    Context
    -------
    n_tasks: int
    n_jobs: int

    Example
    -------
    >>> DeterministicJobAssignment().sample(Random(), n_tasks=10, n_jobs=3)
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]

    """

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return ("n_tasks",)

    @override
    def sample(
        self, rng: Random, *, n_tasks: int, n_jobs: int, **context: Any
    ) -> list[int]:
        q, r = divmod(n_tasks, n_jobs)

        assignments: list[int] = [-1] * n_tasks

        task_id = 0
        for job in range(n_jobs):
            count = q + (1 if job < r else 0)

            for _ in range(count):
                assignments[task_id] = job
                task_id += 1

        return assignments


class Range(Process[list[int]]):
    """Generate a range of integers from 0 to n_tasks-1.

    Context
    -------
    n_tasks: int

    """

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        return ("n_tasks",)

    @override
    def sample(self, rng: Random, *, n_tasks: int, **context: Any) -> list[int]:
        return list(range(n_tasks))

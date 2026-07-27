"""Deterministic processes for scheduling problems."""

from random import Random
from typing import Any

from typing_extensions import override

from cpscheduler.environment.utils.symbols import BaseShapeDim, SymbolicDim
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
    """Generate a range of integers from start to end.

    This function accepts symbolic dimensions in the start and end parameters.

    Context
    -------
    Any symbolic dimensions used.

    """

    start: SymbolicDim
    end: SymbolicDim

    def __init__(
        self, start_or_stop: int | str = 0, stop: int | str | None = None
    ):
        """Initialize the Range process.

        Parameters
        ----------
        start_or_stop: int | str
            The starting value of the range (inclusive) or the stopping value
            if only one argument is provided.

        stop: int | str | None
            The stopping value of the range (exclusive). If None, the range will
            go from start_or_stop to start_or_stop + 1.

        """
        if stop is None:
            stop = start_or_stop
            start_or_stop = 0

        self.start = SymbolicDim.from_shapedim(start_or_stop)
        self.end = SymbolicDim.from_shapedim(stop)

    @property
    @override
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        dim = self.end - self.start

        return (dim.raw,)

    @override
    def sample(self, rng: Random, **context: Any) -> list[int]:
        start = self.start.resolve(**context)
        end = self.end.resolve(**context)

        return list(range(start, end))

    @override
    def __repr__(self) -> str:
        return f"Range({self.start}, {self.end})"

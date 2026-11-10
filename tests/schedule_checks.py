"""Independent checks of simulation results.

Nothing here relies on the propagators or on the objectives: schedules are
checked against the definition of the problem, so a bug in the kernel cannot
hide itself by also corrupting the check.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from random import Random
from typing import NamedTuple, TypeAlias

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    Makespan,
    OptionalityConstraint,
    SchedulingEnv,
)
from cpscheduler.environment.constants import Int
from cpscheduler.environment.constraints import Constraint
from cpscheduler.environment.state import ScheduleState


class TaskSchedule(NamedTuple):
    machine: int
    start: int
    end: int


Schedule: TypeAlias = dict[int, TaskSchedule]

BACKENDS = ("des", "step", "tetris", "gt_tetris")


def schedule_of(state: ScheduleState) -> Schedule:
    """Return what the kernel decided, every expectation comes from the test."""
    return {
        task: TaskSchedule(
            state.get_assignment(task),
            state.get_start(task),
            state.get_end(task),
        )
        for task in state.get_assigned_tasks()
    }


def intersects(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def overlap(a: TaskSchedule, b: TaskSchedule) -> bool:
    return intersects(a.start, a.end, b.start, b.end)


def machine_conflicts(schedule: Schedule) -> list[tuple[int, int]]:
    return [
        (i, j)
        for (i, s_i), (j, s_j) in combinations(schedule.items(), 2)
        if s_i.machine == s_j.machine and overlap(s_i, s_j)
    ]


def duration_errors(
    schedule: Schedule, duration: Callable[[int, int], int]
) -> list[int]:
    """Return the tasks whose length differs from `duration(task, machine)`."""
    return [
        task
        for task, (machine, start, end) in schedule.items()
        if end - start != duration(task, machine)
    ]


def job_order_violations(
    schedule: Schedule, jobs: Sequence[int], operations: Sequence[int]
) -> list[int]:
    """Return the operations starting before the previous one of their job ends."""
    ends = {
        (jobs[task], operations[task]): scheduled.end
        for task, scheduled in schedule.items()
    }

    return [
        task
        for task, scheduled in schedule.items()
        if operations[task] > 0
        and scheduled.start < ends[jobs[task], operations[task] - 1]
    ]


def max_concurrency(intervals: Iterable[TaskSchedule]) -> int:
    intervals = list(intervals)

    return max(
        (
            sum(
                other.start <= scheduled.start < other.end
                for other in intervals
            )
            for scheduled in intervals
        ),
        default=0,
    )


def capacity_violations(
    schedule: Schedule, usage: Sequence[int], capacity: int
) -> list[int]:
    """Return the times where the running tasks use more than the capacity."""
    return [
        scheduled.start
        for scheduled in schedule.values()
        if sum(
            usage[task]
            for task, other in schedule.items()
            if other.start <= scheduled.start < other.end
        )
        > capacity
    ]


def calendar_violations(
    schedule: Schedule, unavailable: Mapping[Int, Sequence[tuple[Int, Int]]]
) -> list[int]:
    """Return the tasks running while their machine is unavailable."""
    down_times = {
        int(machine): [(int(down), int(up)) for down, up in intervals]
        for machine, intervals in unavailable.items()
    }

    return [
        task
        for task, (machine, start, end) in schedule.items()
        if any(
            intersects(start, end, down, up)
            for down, up in down_times.get(machine, [])
        )
    ]


def run_random_episode(
    env: SchedulingEnv, rng: Random, max_steps: int = 5000
) -> None:
    """Play actions from the action support until the episode ends."""
    env.reset()

    for _ in range(max_steps):
        support = env.get_action_support()
        action = ("execute", rng.choice(support)) if support else None

        _, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            return

    raise AssertionError(f"Episode did not finish in {max_steps} steps.")


def random_job_shop(
    rng: Random, n_jobs: int, n_machines: int
) -> dict[str, list[int]]:
    jobs: list[int] = []
    operations: list[int] = []
    machines: list[int] = []
    durations: list[int] = []

    for job in range(n_jobs):
        for operation, machine in enumerate(
            rng.sample(range(n_machines), n_machines)
        ):
            jobs.append(job)
            operations.append(operation)
            machines.append(machine)
            durations.append(rng.randint(1, 9))

    return {
        "job": jobs,
        "operation": operations,
        "machine": machines,
        "processing_time": durations,
    }


def optional_parallel_env(
    constraint: Constraint, n_tasks: int
) -> SchedulingEnv:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(n_tasks, disjunctive=False),
        [constraint, OptionalityConstraint()],
        Makespan(),
        instance={"processing_time": [1] * n_tasks},
    )
    env.reset()

    return env

"""Independent checks of simulation results.

Nothing here relies on the propagators or on the objectives: schedules are
checked against the definition of the problem, so a bug in the kernel cannot
hide itself by also corrupting the check.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations, pairwise, permutations, product
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

TRANSITIONS = ("des", "step", "tetris", "gt_tetris")


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


JobShop: TypeAlias = Mapping[str, Sequence[int]]
Starts: TypeAlias = tuple[int, ...]


def _job_predecessors(instance: JobShop) -> list[int | None]:
    previous = {
        (job, operation): task
        for task, (job, operation) in enumerate(
            zip(instance["job"], instance["operation"], strict=True)
        )
    }

    return [
        previous.get((job, operation - 1))
        for job, operation in zip(
            instance["job"], instance["operation"], strict=True
        )
    ]


def _ready_times(starts: Starts, instance: JobShop) -> list[int]:
    durations = instance["processing_time"]

    return [
        0 if parent is None else starts[parent] + durations[parent]
        for parent in _job_predecessors(instance)
    ]


def semi_active_schedules(instance: JobShop) -> set[Starts]:
    """Return every semi-active schedule of a job shop.

    Each feasible choice of machine sequences gives exactly one semi-active
    schedule, where every operation starts as soon as its job and machine
    predecessors end (French, 1982).
    """
    durations = instance["processing_time"]
    job_parent = _job_predecessors(instance)
    n_tasks = len(durations)

    machine_tasks: dict[int, list[int]] = {}
    for task, machine in enumerate(instance["machine"]):
        machine_tasks.setdefault(machine, []).append(task)

    schedules: set[Starts] = set()
    for sequences in product(*map(permutations, machine_tasks.values())):
        parents = [[] if p is None else [p] for p in job_parent]
        for sequence in sequences:
            for before, after in pairwise(sequence):
                parents[after].append(before)

        # A cycle between the job and machine sequences leaves tasks unstarted
        starts: dict[int, int] = {}
        for _ in range(n_tasks):
            for task in range(n_tasks):
                if task not in starts and all(
                    p in starts for p in parents[task]
                ):
                    starts[task] = max(
                        (starts[p] + durations[p] for p in parents[task]),
                        default=0,
                    )

        if len(starts) == n_tasks:
            schedules.add(tuple(starts[task] for task in range(n_tasks)))

    return schedules


def _idle_gaps(
    starts: Starts, instance: JobShop, task: int
) -> list[tuple[int, int]]:
    """Return the idle intervals of the machine of `task`, ignoring `task`."""
    durations = instance["processing_time"]
    machine = instance["machine"][task]

    busy = sorted(
        (starts[other], starts[other] + durations[other])
        for other, other_machine in enumerate(instance["machine"])
        if other_machine == machine and other != task
    )

    gaps: list[tuple[int, int]] = []
    free_from = 0
    for start, end in busy:
        if free_from < start:
            gaps.append((free_from, start))

        free_from = max(free_from, end)

    gaps.append((free_from, MAX_GAP_END))
    return gaps


MAX_GAP_END = 2**62


def is_active(starts: Starts, instance: JobShop) -> bool:
    """Whether no operation can start earlier without delaying another one.

    Moving an operation into an idle gap of its machine, before its current
    start, delays no other operation (Giffler and Thompson, 1960).
    """
    durations = instance["processing_time"]

    for task, ready in enumerate(_ready_times(starts, instance)):
        for gap_start, gap_end in _idle_gaps(starts, instance, task):
            earliest = max(gap_start, ready)

            if (
                earliest < starts[task]
                and earliest + durations[task] <= gap_end
            ):
                return False

    return True


def is_non_delay(starts: Starts, instance: JobShop) -> bool:
    """Whether no machine is kept idle while one of its operations is ready.

    (Baker, 1974)
    """
    for task, ready in enumerate(_ready_times(starts, instance)):
        for gap_start, gap_end in _idle_gaps(starts, instance, task):
            if max(gap_start, ready) < min(gap_end, starts[task]):
                return False

    return True


def reachable_schedules(env: SchedulingEnv) -> set[Starts]:
    """Return the start times of every episode reachable from the masks.

    A decision with an empty mask only lets the clock advance. Each prefix is
    replayed from a fresh reset, so the enumeration does not rely on
    backtracking.
    """
    schedules: set[Starts] = set()
    prefixes: list[list[tuple[str, int, int] | None]] = [[]]

    while prefixes:
        prefix = prefixes.pop()
        env.reset()
        for action in prefix:
            env.step(action)

        state = env.state
        if state.is_terminal():
            schedules.add(
                tuple(state.get_start(task) for task in range(state.n_tasks))
            )
            continue

        mask = env.transition.get_action_mask(state)
        assert mask or len(prefix) < 10 * state.n_tasks, (
            f"Dead end after {prefix}."
        )

        if not mask:
            prefixes.append([*prefix, None])

        prefixes.extend(
            [*prefix, ("execute", task, machine)]
            for machine, tasks in mask.items()
            for task in tasks
        )

    return schedules

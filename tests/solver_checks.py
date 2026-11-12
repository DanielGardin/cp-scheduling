from collections.abc import Callable
from itertools import permutations, product
from random import Random
from typing import TypeAlias

from cpscheduler.environment import (
    Makespan,
    MaximumLateness,
    Objective,
    SchedulingEnv,
    TotalCompletionTime,
    TotalFlowTime,
    TotalTardiness,
    TotalTardyJobs,
    WeightedCompletionTime,
    WeightedTardiness,
    WeightedTardyJobs,
)
from cpscheduler.environment.transition import ActionType

InstanceData: TypeAlias = dict[str, list[int] | list[float]]
EnvFactory: TypeAlias = Callable[[], SchedulingEnv]


def replay(env: SchedulingEnv, action: ActionType) -> float:
    env.reset()
    *_, truncated, _ = env.step(action)

    assert not truncated
    assert env.state.is_terminal()

    return env.objective.value


def brute_force(
    make_env: EnvFactory, n_tasks: int, rejectable: bool = False
) -> float:
    """Best value over every task order and machine, appended to the schedule.

    For regular objectives, the appended schedules contain an optimal one.
    With `rejectable`, every subset of the tasks may also be rejected first.
    """
    best = float("inf")
    n_machines = make_env().instance.n_machines
    subsets = (
        [
            [task for task in range(n_tasks) if mask >> task & 1]
            for mask in range(1 << n_tasks)
        ]
        if rejectable
        else [[]]
    )

    for rejected in subsets:
        kept = [task for task in range(n_tasks) if task not in rejected]

        for order in permutations(kept):
            for machines in product(range(n_machines), repeat=len(kept)):
                env = make_env()
                env.reset()
                env.step([("reject", task) for task in rejected])

                for task, machine in zip(order, machines, strict=True):
                    # Constraints may reject a task this combination keeps
                    if env.state.infeasible or env.state.is_fixed(task):
                        break

                    env.step(("execute", task, machine))

                if not env.state.infeasible and env.state.is_terminal():
                    best = min(best, env.objective.value)

    return best


REGULAR_OBJECTIVES: dict[str, tuple[Callable[[], Objective], list[str]]] = {
    "makespan": (Makespan, []),
    "total_completion": (TotalCompletionTime, []),
    "weighted_completion": (WeightedCompletionTime, ["weight"]),
    "flow_time": (TotalFlowTime, ["release_time"]),
    "max_lateness": (MaximumLateness, ["due_date"]),
    "tardiness": (TotalTardiness, ["due_date"]),
    "weighted_tardiness": (WeightedTardiness, ["due_date", "weight"]),
    "tardy_jobs": (TotalTardyJobs, ["due_date"]),
    "weighted_tardy_jobs": (WeightedTardyJobs, ["due_date", "weight"]),
}


def regular_instance(
    rng: Random, n_tasks: int, features: list[str]
) -> InstanceData:
    data: InstanceData = {
        "processing_time": [rng.randint(1, 9) for _ in range(n_tasks)],
        "due_date": [rng.randint(1, 20) for _ in range(n_tasks)],
        "weight": [float(rng.randint(1, 5)) for _ in range(n_tasks)],
        "release_time": [0] * n_tasks,
    }

    return {key: data[key] for key in ("processing_time", *features)}

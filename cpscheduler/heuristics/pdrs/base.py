import random
from math import log

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import TaskID, Status
from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.environment.des import SingleInstruction

EXECUTING_STATUS = Status.EXECUTING

def prob_to_lmbda(prob: float, size: int, n_iter: int) -> float:
    """
    Convert a probability to a lambda parameter for the Plackett-Luce model.
    """
    if prob == 1.0:
        return float("inf")

    if prob * size < 1:
        raise ValueError(
            f"Target probability {prob} cannot be lower than uniform probability 1/{size}."
        )

    x = 1 - prob
    for _ in range(n_iter):
        x = (prob * x**size * (size - 1) - (1 - prob)) / (
            size * prob * x ** (size - 1) - 1
        )

    return -log(x)

def solve_p_star_temperature(
    priorities: list[float], target_prob: float, n_iter: int
) -> float:
    n = len(priorities)

    ordered_priorities = sorted(priorities, reverse=True)
    
    ts = sum(
        prio * rank
        for rank, prio in enumerate(ordered_priorities)
    )

    lmbda = ts * 12 / (n*(n+1)*(n-1))
    target_lmbda = prob_to_lmbda(target_prob, n, n_iter)

    return lmbda / target_lmbda

def sample_gumbel(rng: random.Random | None) -> float:
    if rng is None:
        u = random.random()

    else:
        u = rng.random()
    
    return -log(-log(u))

def select_task(
    priorities: list[float],
    available_tasks: set[TaskID],
    temperature: float = 0.0,
    *,
    rng: random.Random | None = None
) -> SingleInstruction | None:
    best_prio = float("-inf")

    dispatching_task: TaskID | None = None
    for task_id in sorted(available_tasks):
        prio = priorities[task_id]

        if temperature > 0.0:
            prio += temperature * sample_gumbel(rng)

        if prio > best_prio:
            if prio == float("inf"):
                return ("execute", task_id)

            best_prio = prio
            dispatching_task = task_id

    if dispatching_task is None:
        return None

    return ("execute", dispatching_task)

@mypyc_attr(allow_interpreted_subclasses=True)
class PriorityDispatchingRule:
    def __init__(self, seed: int | None = None):
        self._internal_rng = random.Random(seed)

    def priority_score(self, obs: DefaultObservation) -> list[float]:
        """
        Simple priority function, must return a score for each task in the
        observation.
        Tasks with higher priority values are considered before the ones with
        a lower priority.

        Note: Negative infinite values are interpreted as tasks that are
        impossible to schedule.
        """
        raise NotImplementedError(
            f"The `priority_score` method must be implemented for {type(self)}."
        )

    def __call__(self, obs: DefaultObservation) -> SingleInstruction | None:
        priorities = self.priority_score(obs)

        return select_task(priorities, obs.available_tasks)

    def sample(
        self,
        obs: DefaultObservation,
        *,
        temperature: float = 1.0,
        target_prob: float | None = None,
        n_iter: int = 5
    ) -> SingleInstruction | None:
        priorities = self.priority_score(obs)

        if target_prob is not None:
            available_priorities = [
                priorities[task_id] for task_id in obs.available_tasks
            ]

            temperature = solve_p_star_temperature(
                available_priorities, target_prob, n_iter
            )

        return select_task(
            priorities,
            obs.available_tasks,
            temperature,
            rng=self._internal_rng
        )

    def ranking(
        self, obs: DefaultObservation, strict: bool = False
    )-> list[SingleInstruction]:
        priorities = self.priority_score(obs)
        status = obs.task['status']

        filtered_priorities = sorted([
            (-prio, task_id) for task_id, prio in enumerate(priorities)
            if status[task_id] < EXECUTING_STATUS
        ])

        instruction = "execute" if strict else "submit"

        return [
            (instruction, task_id) for _, task_id in filtered_priorities
        ]

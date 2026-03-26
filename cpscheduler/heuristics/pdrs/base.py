import random
from math import log

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import TaskID, Status
from cpscheduler.environment.state import ObsType
from cpscheduler.environment.des import SingleAction

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


def select_task(masked_priorities: list[float]) -> TaskID | None:
    dispatching_task: TaskID | None = None
    best_prio = float("-inf")

    for task_id, prio in enumerate(masked_priorities):
        if prio > best_prio:
            dispatching_task = task_id

            if prio == float("inf"):
                break

            best_prio = prio
    
    return dispatching_task

@mypyc_attr(allow_interpreted_subclasses=True)
class PriorityDispatchingRule:
    def __init__(self, seed: int | None = None):
        self._internal_rng = random.Random(seed)

    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        """
        Simple priority function, must return a score for each task in the
        observation.
        Tasks with higher priority values are considered before the ones with
        a lower priority.

        Note: Negative infinite values are interpreted as tasks that are
        impossible to schedule.
        """
        raise NotImplementedError(
            f"The `priority_score` method must be implemented for {self.__class__}."
        )

    def __call__(self, obs: ObsType, time: int | None = None) -> SingleAction | None:
        priorities = self.priority_score(obs, time)
        available: list[bool] = obs[0]['available']

        masked_priorities = [
            prio if available[i] else float("-inf")
            for i, prio in enumerate(priorities)
        ]

        dispatching_task = select_task(masked_priorities)
        return None if dispatching_task is None else ("execute", dispatching_task)

    def sample(
        self, obs: ObsType, time: int | None,
        *,
        temperature: float = 1.0,
        target_prob: float | None = None,
        n_iter: int = 5
    ) -> SingleAction | None:
        priorities = self.priority_score(obs, time)
        available: list[bool] = obs[0]['available']

        if target_prob is not None:
            available_priorities = [
                prio for i, prio in enumerate(priorities) if available[i]
            ]

            temperature = solve_p_star_temperature(
                available_priorities, target_prob, n_iter
            )

        masked_priorities = [
            prio + temperature * sample_gumbel(self._internal_rng)
            if available[i] else float("-inf")
            for i, prio in enumerate(priorities)
        ]

        dispatching_task = select_task(masked_priorities)

        return None if dispatching_task is None else ("execute", dispatching_task)

    def ranking(
        self, obs: ObsType, time: int | None = None, strict: bool = False
    )-> list[SingleAction]:
        priorities = self.priority_score(obs, time)
        status = obs[0]['status']

        filtered_priorities = sorted([
            (-prio, task_id) for task_id, prio in enumerate(priorities)
            if status[task_id] < EXECUTING_STATUS
        ])

        instruction = "execute" if strict else "submit"

        return [
            (instruction, task_id) for _, task_id in filtered_priorities
        ]
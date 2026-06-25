"""Base class for priority dispatching rules."""

import random
from math import log
from typing import Literal

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import Status, TaskID
from cpscheduler.environment.des import SingleInstruction
from cpscheduler.environment.observation import DefaultObservation

EXECUTING_STATUS = Status.EXECUTING


def prob_to_lmbda(prob: float, size: int, n_iter: int) -> float:
    """Convert a probability to a lambda parameter for the Plackett-Luce model.

    This method considers, supposing that the probabilities of a Plackett-Luce
    model decay exponentially with the rank, the lambda parameter that would
    yield a probability of selecting the best item equal to `prob` when there
    are `size` items to choose from.
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
    """Automatic temperature selection for the Plackett-Luce model.

    This method computes a correction factor for the logits of the priorities,
    assuming they approximately follow a linear decay with the rank.
    This correction factor is computed such that the probability of selecting the
    best item is approximately equal to `target_prob`.

    Parameters
    ----------
    priorities : list[float]
        List of priority scores for the available tasks.
        The logits of the Plackett-Luce model are proportional to these scores.

    target_prob : float
        The target probability of selecting the best item.

    n_iter : int
        The number of iterations for the numerical method.

    Returns
    -------
    float
        The computed temperature parameter.

    """
    n = len(priorities)

    ordered_priorities = sorted(priorities, reverse=True)

    ts = sum(prio * rank for rank, prio in enumerate(ordered_priorities))

    lmbda = ts * 12 / (n * (n + 1) * (n - 1))
    target_lmbda = prob_to_lmbda(target_prob, n, n_iter)

    return lmbda / target_lmbda


def select_task(
    priorities: list[float],
    available_tasks: set[TaskID],
    temperature: float = 0.0,
    *,
    rng: random.Random | None = None,
) -> SingleInstruction | None:
    """Select a task to dispatch based on the given priorities.

    This method uses the Gumbel-max trick to sample from a Plackett-Luce model
    defined by the given priorities. The selected task is the available task
    with the highest perturbed priority score.
    """
    if rng is None:
        rng = random.Random()

    best_prio = float("-inf")

    dispatching_task: TaskID | None = None
    for task_id in sorted(available_tasks):
        prio = priorities[task_id]

        if temperature > 0.0:
            # Gumbel-max trick: add Gumbel noise to the priority score
            prio += temperature * -log(-log(rng.random()))

        if prio > best_prio:
            if prio == float("inf"):
                return ("execute", task_id)

            best_prio = prio
            dispatching_task = task_id

    if dispatching_task is None:
        return None

    return ("execute", dispatching_task)


SCHEDULE_GENERATION_METHODS = Literal["serial", "parallel"]


@mypyc_attr(allow_interpreted_subclasses=True)
class PriorityDispatchingRule:
    """Base class for priority dispatching rules.

    This is a simple class that defines the interface for priority dispatching
    rules.
    A priority dispatching rule (PDR) is a function that assigns a priority
    score to each task in the environment, and executes the task with the
    highest priority score among the available tasks.

    The priority score can be any real number, and the PDR can be deterministic
    or stochastic. In case of stochastic PDRs, the task is selected with a probability
    proportional to its priority score, using a Softmax (or Plackett-Luce) model.

    Note that this implementation does not select the machine to execute the task,
    as this is assumes a list scheduling machine dispatching policy.
    For multi-machine dispatching, the PDR can be combined with a machine selection
    policy, or the PDR can be extended to select both the task and the machine.

    See Also
    --------
    FUTURE: Add reference to extensions of this class that implement multi-machine PDRs.

    """

    _internal_rng: random.Random

    def __init__(self, seed: int | None = None):
        """Initialize the priority dispatching rule."""
        self._internal_rng = random.Random(seed)

    def priority_score(self, obs: DefaultObservation) -> list[float]:
        """Priority score function for the dispatching rule.

        This method assigns a priority score to each task in the environment,
        based on the current observation.
        This mapping can use cross-task information, and other information
        from the observation, such as the current time, the status of the tasks,
        and the status of the machines.

        The value -infinity can be used to indicate that a task cannot be
        executed, and will be ignored by the dispatching rule.
        Likewise, the value +infinity indicates that a task must be executed,
        and will be selected by the dispatching rule if it is available.

        Parameters
        ----------
        obs : DefaultObservation
            The current observation of the environment.

        Returns
        -------
        list[float]
            A list of priorities with lenght `n_tasks`.
        """
        raise NotImplementedError(
            f"The `priority_score` method must be implemented for {type(self)}."
        )

    # Internal method to compute the priority score, can be overridden by subclasses
    # to implement custom behavior.
    def _priority_score(self, obs: DefaultObservation) -> list[float]:
        return self.priority_score(obs)

    def __call__(self, obs: DefaultObservation) -> SingleInstruction | None:
        """Deterministically select a task to dispatch based on the current observation."""
        priorities = self._priority_score(obs)

        return select_task(priorities, obs.available_tasks)

    def sample(
        self,
        obs: DefaultObservation,
        *,
        temperature: float = 1.0,
        target_prob: float | None = None,
        n_iter: int = 5,
    ) -> SingleInstruction | None:
        """Stochastically select a task to dispatch based on the current observation.

        This method uses a softmax model to select a task to dispatch.

        Parameters
        ----------
        obs : DefaultObservation
            The current observation of the environment.

        temperature : float, optional
            The temperature parameter for the softmax model.
            A higher temperature results in a more uniform distribution, while
            a lower temperature results in a more greedy selection of the task
            with the highest priority score. Default is 1.0.

        target_prob : float | None, optional
            If not None, the temperature parameter is automatically adjusted such that
            the probability of selecting the task with the highest priority score is
            approximately equal to `target_prob`. Default is None.

        n_iter : int, optional
            The number of iterations for the numerical method used to compute the
            temperature parameter. Default is 5.

        Returns
        -------
        SingleInstruction | None
            A tuple of the form ("execute", task_id) if a task is selected, or None
            if no task is available to dispatch.

        """
        priorities = self._priority_score(obs)

        if target_prob is not None:
            available_priorities = [
                priorities[task_id] for task_id in obs.available_tasks
            ]

            temperature = solve_p_star_temperature(
                available_priorities, target_prob, n_iter
            )

        return select_task(
            priorities, obs.available_tasks, temperature, rng=self._internal_rng
        )

    def ranking(
        self,
        obs: DefaultObservation,
        schedule_generation: SCHEDULE_GENERATION_METHODS,
    ) -> list[SingleInstruction]:
        """Return a ranking of the available tasks based on their priority scores.

        This method returns a list of tasks sorted by their priority scores,
        in descending order.
        This permutation can be decoded into a schedule by the environment, using
        either a serial or parallel schedule generation scheme.

        Parameters
        ----------
        obs : DefaultObservation
            The current observation of the environment.

        schedule_generation : SCHEDULE_GENERATION_METHODS
            The schedule generation scheme to use for decoding the ranking
            into a schedule.
            The available options are:
            - "serial": The tasks are executed one at a time, in the order of the
                ranking.
            - "parallel": The tasks are executed in parallel, in the order of the
                ranking, until all available machines are busy.

        """
        priorities = self._priority_score(obs)

        if schedule_generation == "parallel":
            filtered_priorities = sorted(
                [(-prio, task_id) for task_id, prio in enumerate(priorities)]
            )

            return [("submit", task_id) for _, task_id in filtered_priorities]

        if schedule_generation == "serial":
            status = obs.task["status"]
            filtered_priorities = sorted(
                [
                    (-prio, task_id)
                    for task_id, prio in enumerate(priorities)
                    if status[task_id] < EXECUTING_STATUS
                ]
            )

            return [("execute", task_id) for _, task_id in filtered_priorities]

        raise ValueError(
            f"Invalid schedule generation method: {schedule_generation}. "
            f"Expected one of: 'serial', 'parallel'."
        )

    def sample_ranking(
        self,
        obs: DefaultObservation,
        schedule_generation: SCHEDULE_GENERATION_METHODS,
        *,
        temperature: float = 1.0,
        target_prob: float | None = None,
        n_iter: int = 5,
    ) -> list[SingleInstruction]:
        """Return a stochastically sampled ranking of the available tasks.

        This method returns a list of tasks, sampled from a Plackett-Luce model
        defined by the priority scores, in descending order.
        This permutation can be decoded into a schedule by the environment, using
        either a serial or parallel schedule generation scheme.

        Parameters
        ----------
        obs : DefaultObservation
            The current observation of the environment.

        schedule_generation : SCHEDULE_GENERATION_METHODS
            The schedule generation scheme to use for decoding the ranking
            into a schedule.
            The available options are:
            - "serial": The tasks are executed one at a time, in the order of the
                ranking.
            - "parallel": The tasks are executed in parallel, in the order of the
                ranking, until all available machines are busy.

        temperature : float, optional
            The temperature parameter for the softmax model.
            A higher temperature results in a more uniform distribution, while
            a lower temperature results in a more greedy selection of the task
            with the highest priority score. Default is 1.0.

        target_prob : float | None, optional
            If not None, the temperature parameter is automatically adjusted such that
            the probability of selecting the task with the highest priority score is
            approximately equal to `target_prob`. Default is None.

        n_iter : int, optional
            The number of iterations for the numerical method used to compute the
            temperature parameter. Default is 5.

        """
        priorities = self._priority_score(obs)

        if target_prob is not None:
            available_priorities = [
                priorities[task_id] for task_id in obs.available_tasks
            ]

            temperature = solve_p_star_temperature(
                available_priorities, target_prob, n_iter
            )

        rng = self._internal_rng

        if schedule_generation == "parallel":
            filtered_priorities = sorted(
                [
                    (-prio + temperature * -log(-log(rng.random())), task_id)
                    for task_id, prio in enumerate(priorities)
                ]
            )

            return [("submit", task_id) for _, task_id in filtered_priorities]

        if schedule_generation == "serial":
            status = obs.task["status"]

            filtered_priorities = sorted(
                [
                    (-prio + temperature * -log(-log(rng.random())), task_id)
                    for task_id, prio in enumerate(priorities)
                    if status[task_id] < EXECUTING_STATUS
                ]
            )

            return [("execute", task_id) for _, task_id in filtered_priorities]

        raise ValueError(
            f"Invalid schedule generation method: {schedule_generation}. "
            f"Expected one of: 'serial', 'parallel'."
        )


class StaticPriorityDispatchingRule(PriorityDispatchingRule):
    """Base class for static priority dispatching rules.

    A static priority dispatching rule (PDR) is a function that assigns a
    fixed priority score to each task in the environment, and executes the
    task with the highest priority score among the available tasks.
    The priority scores are computed once at the beginning of the scheduling
    process, and do not change during the scheduling process.
    """

    _current_fingerprint: int | None = None
    _priorities: list[float]

    def initialize(self, obs: DefaultObservation) -> None:
        """Initialize the static priority dispatching rule.

        This method computes the fixed priority scores for each task in the
        environment, based on the initial observation.
        """
        self._priorities = self.priority_score(obs)
        self._current_fingerprint = obs.fingerprint

    def _priority_score(self, obs: DefaultObservation) -> list[float]:
        if self._current_fingerprint != obs.fingerprint:
            self.initialize(obs)

        return self._priorities

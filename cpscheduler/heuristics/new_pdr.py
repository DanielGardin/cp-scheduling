from typing import Any, Generic, TypeVar, TypeAlias
from collections.abc import Iterable, Sequence

import random

from abc import ABC, abstractmethod
from mypy_extensions import mypyc_attr

from cpscheduler.environment._common import Status, ObsType
from cpscheduler.environment.instructions import SingleAction

from ._protocols import ArrayLike, TabularRepresentation
from .list_wrapper import wrap_observation


def filter_tasks(
    obs: TabularRepresentation[ArrayLike], status: Status
) -> TabularRepresentation[ArrayLike]:
    """
    Filters the tasks in the observation based on their status.
    """
    if isinstance(obs, dict):
        return {
            k: v for k, v in obs.items() if k != status and isinstance(v, ArrayLike)
        }

    if isinstance(obs, ArrayLike):
        new_obs: TabularRepresentation[ArrayLike] = obs[obs[status] < Status.COMPLETED]

    raise TypeError(f"Unsupported observation type: {type(obs)}")


@mypyc_attr(allow_interpreted_subclasses=True)
class PriorityDispatchingRule(ABC):
    """
    Abstract class for Priority Dispatching Rule-based policies. To implement one, inherit from this class and implement
    the `priority_rule` method, addressing a priority value for each task. The tasks will be sorted in descending order
    of priority.

    The workings of the policy are based in tabular observations, where the tasks are stored in a list-like structure.
    For this reason, the observations are wrapped into a ArrayLike structure similar to pandas DataFrames, but more
    generally applicable to any structured.
    """

    def __init__(
        self, status: Any = "status", strict: bool = False, job_oriented: bool = False
    ) -> None:
        self.status = status
        self.job_oriented = job_oriented

        self.instruction = "execute" if strict else "submit"
        if job_oriented:
            self.instruction += " job"

    @abstractmethod
    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        """
        Implements a priority rule to sort the tasks in the waiting buffer by a given criterion.
        """
        raise NotImplementedError

    def __call__(self, obs: Any, time: int | None = None) -> Sequence[SingleAction]:
        filtered_obs = filter_tasks(wrap_observation(obs), self.status)

        if time is None:
            time = 0

        priorities = self.priority_rule(filtered_obs, time)
        order = (-priorities).argsort()

        action = [(self.instruction, int(task_id)) for task_id in order]

        return action

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CombinedRule(PriorityDispatchingRule):
    """
    Combined Rule heuristic.

    This heuristic combines multiple dispatching rules to select the next job to be scheduled.
    """

    def __init__(
        self,
        rules: Iterable[PriorityDispatchingRule],
        weights: Iterable[float] | None = None,
        status: Any = "status",
        strict: bool = False,
        job_oriented: bool = False,
    ) -> None:
        super().__init__(status, strict, job_oriented)
        self.rules = list(rules)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.rules)

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        """
        Combines the priority rules of the individual dispatching rules using the specified weights.
        """
        priorities = self.weights[0] * self.rules[0].priority_rule(obs, time)

        for rule, weight in zip(self.rules[1:], self.weights[1:]):
            priorities += weight * rule.priority_rule(obs, time)

        return priorities


class ShortestProcessingTime(PriorityDispatchingRule):
    """
    Shortest Processing Time (SPT) heuristic.

    This heuristic selects the job with the shortest processing time as the next job to be scheduled.
    """

    def __init__(
        self,
        status: Any = "status",
        processing_time: Any = "processing_time",
        strict: bool = False,
        job_oriented: bool = False,
    ) -> None:
        super().__init__(status, strict, job_oriented)
        self.processing_time = processing_time

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        return -obs[self.processing_time]

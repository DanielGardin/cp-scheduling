from typing import Any, TypeAlias, Final, overload
from collections.abc import Iterable, Sequence
from typing_extensions import TypedDict, Unpack

import math

from abc import ABC, abstractmethod
from mypy_extensions import mypyc_attr

from cpscheduler.environment._common import Status, ObsType
from cpscheduler.environment.instructions import SingleAction

from ._protocols import ArrayLike, TabularRepresentation
from .list_wrapper import ListWrapper
from .array_utils import (
    NUMPY_AVAILABLE,
    TORCH_AVAILABLE,
    wrap_observation,
    maximum,
    minimum,
    exp,
    argsort,
    where,
    array_sum,
    array_mean,
    array_max,
    astype,
)

if NUMPY_AVAILABLE:
    import numpy as np

if TORCH_AVAILABLE:
    import torch


def sample_gumbel(x: ArrayLike, seed: int | None = None) -> ArrayLike:
    result: ArrayLike
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        if seed is not None:
            torch.manual_seed(seed)

        result = -torch.log(-torch.log(torch.rand_like(x)))

    elif NUMPY_AVAILABLE:
        if seed is not None:
            np.random.seed(seed)

        result = -np.log(-np.log(np.random.rand(*x.shape)))

    else:
        raise RuntimeError("Gumbel sampling requires either PyTorch or NumPy.")

    return result


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

    return -math.log(x)


def solve_p_star(priorities: ArrayLike, target_prob: float, n_iter: int) -> ArrayLike:
    *batch, n_tasks = priorities.shape
    target_lmbda = prob_to_lmbda(target_prob, n_tasks, n_iter)

    lmbda: ArrayLike
    if TORCH_AVAILABLE and isinstance(priorities, torch.Tensor):
        with torch.no_grad():
            *batch, n_tasks = priorities.shape

            if target_prob * n_tasks < 1:
                raise ValueError(
                    f"Target probability {target_prob} cannot be lower than uniform probability 1/{n_tasks}."
                )

            batch_size = int(torch.prod(torch.tensor(batch)))
            target_lmbda = prob_to_lmbda(target_prob, n_tasks, n_iter)

            X = torch.arange(n_tasks, device=priorities.device, dtype=priorities.dtype)
            X_mean = X.mean()

            ordered_priorities, _ = torch.sort(priorities, dim=-1, descending=True)
            y = priorities.reshape(batch_size, n_tasks)
            y_mean = y.mean(axis=-1)

            cov_xy = torch.sum((X - X_mean) * (y - y_mean), axis=-1)
            var_x = torch.sum((X - X_mean) ** 2, axis=-1)

            lmbda = torch.reshape(-cov_xy / var_x, batch)

    elif NUMPY_AVAILABLE:
        batch_size = int(np.prod(batch))

        x = np.arange(n_tasks)
        x_mean = x.mean()

        ordered_priorities = np.sort(priorities, axis=-1)[..., ::-1]
        y = ordered_priorities.reshape(batch_size, n_tasks)
        y_mean = y.mean(axis=-1)

        cov_xy = np.sum((x - x_mean) * (y - y_mean), axis=-1)
        var_x = np.sum((x - x_mean) ** 2, axis=-1)

        lmbda = np.reshape(-cov_xy / var_x, batch)

    else:
        ordered_priorities = ListWrapper.sort(priorities, reverse=True, stable=True)
        ts = (n_tasks + 1) / 2 - ListWrapper(range(n_tasks))

        lmbda = (
            12
            / (n_tasks * (n_tasks + 1) * (n_tasks - 1))
            * (ts * ordered_priorities).sum()
        )

    return lmbda / target_lmbda


FeatureTag: TypeAlias = str | int


class BasePriorityKwargs(TypedDict, total=False):
    status: FeatureTag | None
    """
    Status" feature tag, if provided, the output action will not include tasks
    in non-executable status. When `None`, all tasks will be considered.
    """

    strict: bool
    """
    If True, the output action is strictly to execute the tasks in the given order.
    Alternatively, the output action will be to submit the tasks to be executed whenever
    the task is available, following the order given by the priority rule.
    """

    job_oriented: bool
    """
    If True, the output action will refer to jobs instead of tasks.
    """


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
        self,
        status: FeatureTag | None = "status",
        strict: bool = False,
        job_oriented: bool = False,
        **kwargs: Any,
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

    def get_priorities(self, obs: Any, time: int | None = None) -> ArrayLike:
        array_obs = wrap_observation(obs)

        if time is None:
            time = 0

        priorities = self.priority_rule(array_obs, time)
        priorities = astype(priorities, float)

        if self.status is not None:
            mask = array_obs[self.status] >= Status.EXECUTING

            priorities[mask] = float("-inf")

        return priorities

    @overload
    def __call__(
        self, obs: TabularRepresentation[ArrayLike], time: int | None = None
    ) -> Sequence[Sequence[SingleAction]]: ...

    @overload
    def __call__(
        self, obs: ArrayLike, time: int | None = None
    ) -> Sequence[Sequence[SingleAction]]: ...

    @overload
    def __call__(
        self, obs: ObsType, time: int | None = None
    ) -> Sequence[SingleAction]: ...

    @overload
    def __call__(
        self, obs: dict[str, list[Any]], time: int | None = None
    ) -> Sequence[SingleAction]: ...

    def __call__(
        self, obs: Any, time: int | None = None
    ) -> Sequence[SingleAction] | Sequence[Sequence[SingleAction]]:
        priorities = self.get_priorities(obs, time)
        order = argsort(priorities, descending=True, stable=True, axis=-1)

        *batch, n_tasks = order.shape

        if len(batch) == 0:
            return [(self.instruction, int(task_id)) for task_id in order]

        return [
            [(self.instruction, int(task_id)) for task_id in order[batch_index]]
            for batch_index in range(len(order))
        ]

    def sample(
        self,
        obs: Any,
        time: int | None = None,
        temp: float | ArrayLike = 1.0,
        target_prob: float | None = None,
        n_iter: int = 5,
        seed: int | None = None,
    ) -> Sequence[SingleAction] | Sequence[Sequence[SingleAction]]:
        priorities = self.get_priorities(obs, time)

        if temp <= 0.0 or (target_prob is not None and target_prob >= 1.0):
            temp = 0.0

        elif target_prob is not None:
            temp = solve_p_star(priorities, target_prob, n_iter)

        priorities = priorities + temp * sample_gumbel(priorities, seed)

        order = argsort(priorities, descending=True, stable=True, axis=-1)

        *batch, n_tasks = order.shape

        if len(batch) == 0:
            return [(self.instruction, int(task_id)) for task_id in order]

        return [
            [(self.instruction, int(task_id)) for task_id in order[batch_index]]
            for batch_index in range(len(order))
        ]

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
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
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
        processing_time: FeatureTag = "processing_time",
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.processing_time = processing_time

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        return -obs[self.processing_time]


class EarliestDueDate(PriorityDispatchingRule):
    """
    Earliest Due Date (EDD) heuristic.

    This heuristic selects the job with the earliest due date as the next job to be scheduled.
    """

    def __init__(
        self, due_date: FeatureTag = "due_date", **kwargs: Unpack[BasePriorityKwargs]
    ) -> None:
        super().__init__(**kwargs)
        self.due_date = due_date

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        return -obs[self.due_date]


class ModifiedDueDate(PriorityDispatchingRule):
    """
    Modified Due Date (MDD) heuristic.
    """

    def __init__(
        self,
        due_date: FeatureTag = "due_date",
        processing_time: FeatureTag = "processing_time",
        weight: FeatureTag | None = None,
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.due_date = due_date
        self.processing_time = processing_time
        self.weight = weight

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        task_dues = maximum(time + obs[self.processing_time], obs[self.due_date])

        if self.weight is not None:
            task_dues = task_dues / obs[self.weight]

        return -task_dues


class WeightedShortestProcessingTime(PriorityDispatchingRule):
    """
    Weighted Shortest Processing Time (WSPT) heuristic.

    This heuristic selects the job with the shortest processing time as the next job to be scheduled, but the processing
    time is weighted by a given factor. It is optimal for 1||sum w_j C_j.
    """

    def __init__(
        self,
        processing_time: FeatureTag = "processing_time",
        weight: FeatureTag = "weight",
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.processing_time = processing_time
        self.weight = weight

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        return obs[self.weight] / obs[self.processing_time]


class MinimumSlackTime(PriorityDispatchingRule):
    """
    Minimum Slack Time (MST) heuristic.

    This heuristic selects the job with the smallest slack time as the next job to be scheduled.
    """

    def __init__(
        self,
        due_date: FeatureTag = "due_date",
        processing_time: FeatureTag = "processing_time",
        release_time: FeatureTag | None = None,
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.due_date = due_date
        self.processing_time = processing_time
        self.release_time = release_time

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        end_times = obs[self.processing_time] + (
            maximum(obs[self.release_time], time)
            if self.release_time is not None
            else time
        )

        slack = end_times - obs[self.due_date]
        return slack


class CriticalRatio(PriorityDispatchingRule):
    """
    Critical Ratio (CR) heuristic.

    This heuristic selects the job with the smallest critical ratio as the next job to be scheduled.
    """

    def __init__(
        self,
        due_date: FeatureTag = "due_date",
        processing_time: FeatureTag = "processing_time",
        release_time: FeatureTag | None = None,
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.due_date = due_date
        self.processing_time = processing_time
        self.release_time = release_time

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        time_window = obs[self.due_date] - (
            maximum(obs[self.release_time], time)
            if self.release_time is not None
            else time
        )

        return time_window / obs[self.processing_time]


class FirstInFirstOut(PriorityDispatchingRule):
    """
    First In First Out (FIFO) heuristic.

    This heuristic selects the job that has been in the waiting buffer the longest as the next job to be scheduled.
    """

    def __init__(
        self,
        release_time: FeatureTag = "release_time",
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.release_time = release_time

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        return time - obs[self.release_time]


class CostOverTime(PriorityDispatchingRule):
    """
    Cost OVER Time (CoverT) heuristic.
    This heuristic assign a cost value to each task
    """

    def __init__(
        self,
        processing_time: FeatureTag = "processing_time",
        due_date: FeatureTag = "due_date",
        release_time: FeatureTag | None = None,
        weight: FeatureTag = "weight",
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.processing_time = processing_time
        self.weight = weight
        self.due_date = due_date
        self.release_time = release_time

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        wspt = obs[self.weight] / obs[self.processing_time]

        sum_processing = array_sum(obs[self.processing_time], axis=-1)

        end_times = obs[self.processing_time] + (
            maximum(obs[self.release_time], time)
            if self.release_time is not None
            else time
        )

        deadline_slack = maximum(sum_processing - obs[self.due_date], 0)

        return wspt * minimum(deadline_slack / (sum_processing - end_times), 1.0)


class ApparentTardinessCost(PriorityDispatchingRule):
    """
    Modified Apparent Tardiness Cost (ATC) heuristic.
    This heuristic assign a cost value to each task
    """

    def __init__(
        self,
        lookahead: float = 3.0,
        processing_time: FeatureTag = "processing_time",
        due_date: FeatureTag = "due_date",
        release_time: FeatureTag | None = None,
        weight: FeatureTag = "weight",
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.processing_time = processing_time
        self.weight = weight
        self.due_date = due_date
        self.release_time = release_time

        self.lookahead = lookahead

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        P_mean = array_mean(obs[self.processing_time], axis=-1)

        wspt = obs[self.weight] / obs[self.processing_time]

        start_time = (
            maximum(obs[self.release_time], time)
            if self.release_time is not None
            else time
        )

        slack = obs[self.due_date] - obs[self.processing_time] - start_time

        return wspt * exp(-maximum(0, slack) / (P_mean * self.lookahead))


class TrafficPriority(PriorityDispatchingRule):
    """
    Traffic Priority (TP) heuristic.
    This heuristic assign a cost value to each task
    """

    def __init__(
        self,
        K: float = 3.0,
        processing_time: FeatureTag = "processing_time",
        due_date: FeatureTag = "due_date",
        # weight: FeatureTag | None = None,
        **kwargs: Unpack[BasePriorityKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.processing_time = processing_time
        self.due_date = due_date
        # self.weight = weight
        self.K = K

    def priority_rule(
        self, obs: TabularRepresentation[ArrayLike], time: int
    ) -> ArrayLike:
        traffic_congestion_ratio = array_sum(
            obs[self.processing_time], axis=-1
        ) / array_mean(obs[self.due_date], axis=-1)

        weighted_edd = self.K / traffic_congestion_ratio - 0.5
        weighted_edd = where(
            weighted_edd < 0.0, 0.0, where(weighted_edd > 1.0, 1.0, weighted_edd)
        )

        max_due_date = array_max(obs[self.due_date], axis=-1)
        max_processing_time = array_max(obs[self.processing_time], axis=-1)

        tp: ArrayLike = -(
            weighted_edd * obs[self.due_date] / max_due_date
            + (1 - weighted_edd) * obs[self.processing_time] / max_processing_time
        )

        return tp

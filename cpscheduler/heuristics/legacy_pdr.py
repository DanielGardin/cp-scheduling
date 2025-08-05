from typing import Any
from collections.abc import Iterable, Sequence

from math import log, exp
import random

from abc import ABC, abstractmethod
from mypy_extensions import mypyc_attr

from cpscheduler.environment._common import ObsType, Status
from cpscheduler.environment.utils import convert_to_list
from cpscheduler.environment.instructions import SingleAction


def sample_gumbel() -> float:
    """Sample from Gumbel(0, 1) using inverse transform sampling."""
    u = random.random()
    return -log(-log(u))


@mypyc_attr(allow_interpreted_subclasses=True)
class PriorityDispatchingRule(ABC):
    """
    Abstract class for Priority Dispatching Rule-based policies. To implement one, inherit from this class and implement
    the `priority_rule` method, addressing a priority value for each task. The tasks will be sorted in descending order
    of priority.
    """

    def __init__(self, strict: bool = False):
        self.instruction = "execute" if strict else "submit"

    @abstractmethod
    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        """
        Implements a priority rule to sort the tasks in the waiting buffer by a given criterion.
        """
        raise NotImplementedError

    def filter_tasks(self, tasks: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {
            feature: [
                value
                for value, status in zip(values, tasks["status"])
                if status < Status.COMPLETED
            ]
            for feature, values in tasks.items()
        }

    def get_priority(
        self, obs: ObsType, current_time: int | None = None
    ) -> list[tuple[float, int]]:
        """
        Get the priority of tasks in the waiting buffer.

        Parameters:
        - obs: tuple[dict[str, list[Any]], dict[str, list[Any]]] (task_state, job_state)
            The current observation of the environment.
        """
        tasks, jobs = obs

        filtered_tasks = self.filter_tasks(tasks)

        if current_time is None:
            current_time = 0

        priority_values = self.priority_rule(filtered_tasks, jobs, current_time)
        task_ids: list[int] = filtered_tasks["task_id"]

        return [
            (priority, task_id) for task_id, priority in zip(task_ids, priority_values)
        ]

    def get_data(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], feature: str
    ) -> list[Any]:
        if feature in tasks:
            return tasks[feature]

        if feature in jobs:
            job_data = jobs[feature]
            return [job_data[job_id] for job_id in tasks["job_id"]]

        raise ValueError(f"Feature '{feature}' not found in tasks or jobs.")

    def __call__(
        self, obs: ObsType, current_time: int | None = None
    ) -> Sequence[SingleAction]:
        priorities = self.get_priority(obs, current_time)

        action = [
            ("submit", task_id)
            for _, task_id in sorted(priorities, key=lambda x: -x[0])
        ]

        return action

    def sample(
        self,
        obs: ObsType,
        current_time: int | None = None,
        temp: float = 1.0,
        target_prob: float | None = None,
        n_iter: int = 5,
        seed: int | None = None,
    ) -> Sequence[SingleAction]:
        """
        Sample a task based on the priority rule. Instead of greedily selecting the task with the highest priority,
        this method uses a Plackett-Luce model to sample a task based on the priority values.

        Parameters:
        - obs: tuple[dict[str, list[Any]], dict[str, list[Any]]] (task_state, job_state)
            The current observation of the environment.

        - temp: float, default=1.0
            The temperature parameter for the Plackett-Luce model. Higher values make the sampling more uniform,
            while lower values make it more greedy. Temp = 0.0 is equivalent to greedy sampling.
        """
        priorities = self.get_priority(obs, current_time)

        if temp <= 0.0 or (target_prob is not None and target_prob >= 1.0):
            temp = 0.0

        elif target_prob is not None:
            n = len(priorities)

            predicted_lambda = (
                12
                / (n * (n + 1) * (n - 1))
                * sum(
                    [
                        logit * ((n + 1) / 2 - k)
                        for k, (logit, _) in enumerate(
                            sorted(priorities, key=lambda x: -x[0]), 1
                        )
                    ]
                )
            )

            x = 1 - target_prob

            for _ in range(n_iter):
                x = (target_prob * x**n * (n - 1) - (1 - target_prob)) / (
                    n * target_prob * x ** (n - 1) - 1
                )

            temp = predicted_lambda / -log(x)

        if seed is not None:
            random.seed(seed)

        priorities = sorted(
            [
                (-priority + temp * sample_gumbel(), task_id)
                for (priority, task_id) in priorities
            ]
        )

        action = [("submit", task_id) for _, task_id in priorities]

        return action


class CombinedRule(PriorityDispatchingRule):
    """
    Combined Rule heuristic.

    This heuristic combines multiple dispatching rules to select the next job to be scheduled.
    """

    def __init__(
        self,
        rules: Iterable[PriorityDispatchingRule],
        weights: Iterable[float] | None = None,
        strict: bool = False,
    ):
        super().__init__(strict)
        self.rules = list(rules)
        self.weights = (
            convert_to_list(weights, float)
            if weights is not None
            else [1.0] * len(self.rules)
        )

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        priority_values = [rule.priority_rule(tasks, jobs, time) for rule in self.rules]

        return [
            sum(weight * value for weight, value in zip(self.weights, values))
            for values in zip(*priority_values)
        ]


class ShortestProcessingTime(PriorityDispatchingRule):
    """
    Shortest Processing Time (SPT) heuristic.

    This heuristic selects the job with the shortest processing time as the next job to be scheduled.
    """

    def __init__(
        self, processing_time_label: str = "processing_time", strict: bool = False
    ):
        super().__init__(strict)
        self.processing_time_label = processing_time_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        processing_times: list[int] = self.get_data(
            tasks, jobs, self.processing_time_label
        )

        return [-processing_time for processing_time in processing_times]


class MostOperationsRemaining(PriorityDispatchingRule):
    """
    Most Operations Remaining (MOPNR) heuristic.

    This heuristic selects the earliest job to be done in the waiting buffer as the next job to be scheduled.
    """

    def __init__(self, operation_label: str = "operation", strict: bool = False):
        super().__init__(strict)
        self.operation_label = operation_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        operation: list[int] = self.get_data(tasks, jobs, self.operation_label).copy()
        job_ids: list[int] = tasks["job_id"]

        max_ops: list[int] = [0 for _ in range(len(job_ids))]
        for job, op in zip(job_ids, operation):
            max_ops[job] = max(max_ops[job], op)

        return [max_ops[job] - op for job, op in zip(tasks["job_id"], operation)]


class RandomPriority(PriorityDispatchingRule):
    """
    Random Priority heuristic.

    This heuristic randomly selects a job from the waiting buffer as the next job to be scheduled.
    """

    def __init__(self, seed: int | None = None, strict: bool = False):
        super().__init__(strict)
        self.seed = seed

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        if self.seed is not None:
            random.seed(self.seed)

        return [random.random() for _ in range(len(tasks["task_id"]))]


class MostWorkRemaining(PriorityDispatchingRule):
    """
    Most Work Remaining (MWKR) heuristic.

    This heuristic selects the job with the most work remaining as the next job to be scheduled.
    """

    def __init__(
        self,
        operation_label: str = "operation",
        processing_time_label: str = "processing_time",
        strict: bool = False,
    ):
        super().__init__(strict)
        self.operation_label = operation_label
        self.processing_time_label = processing_time_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        ops: list[int] = self.get_data(tasks, jobs, self.operation_label)
        procs: list[int] = self.get_data(tasks, jobs, self.processing_time_label)
        job_ids: list[int] = tasks["job_id"]

        cumulative_processing_times: list[list[float]] = [
            [] for _ in range(len(jobs["job_id"]))
        ]

        for task_id, job_id in enumerate(job_ids):
            cumulative_processing_times[job_id].append(0.0)

        for task_id, job_id in enumerate(job_ids):
            op_number = ops[task_id]
            proc_time = procs[task_id]

            for i in range(op_number + 1):
                cumulative_processing_times[job_id][i] += proc_time

        return [
            cumulative_processing_times[job][op]
            for job, op in zip(tasks["job_id"], ops)
        ]


class EarliestDueDate(PriorityDispatchingRule):
    """
    Earliest Due Date (EDD) heuristic.

    This heuristic selects the job with the earliest due date as the next job to be scheduled.
    """

    def __init__(self, due_date_label: str = "due_date", strict: bool = False):
        super().__init__(strict)
        self.due_date_label = due_date_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        due_dates: list[int] = self.get_data(tasks, jobs, self.due_date_label)

        return [-due_date for due_date in due_dates]


class ModifiedDueDate(PriorityDispatchingRule):
    """
    Modified Due Date (MDD) heuristic.
    """

    def __init__(
        self,
        due_date_label: str = "due_date",
        processing_time_label: str = "processing_time",
        weight_label: str | None = None,
        strict: bool = False,
    ):
        super().__init__(strict)
        self.due_date_label = due_date_label
        self.processing_time_label = processing_time_label
        self.weight_label = weight_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        due_dates: list[int] = self.get_data(tasks, jobs, self.due_date_label)
        processing_times: list[int] = self.get_data(
            tasks, jobs, self.processing_time_label
        )

        if self.weight_label is not None:
            weights: list[float] = self.get_data(tasks, jobs, self.weight_label)

            return [
                -max(proc_time + time, due_date) / weight
                for proc_time, due_date, weight in zip(
                    processing_times, due_dates, weights
                )
            ]

        return [
            -max(proc_time + time, due_date)
            for proc_time, due_date in zip(processing_times, due_dates)
        ]


class WeightedShortestProcessingTime(PriorityDispatchingRule):
    """
    Weighted Shortest Processing Time (WSPT) heuristic.

    This heuristic selects the job with the shortest processing time as the next job to be scheduled, but the processing
    time is weighted by a given factor. It is optimal for 1||sum w_j C_j.
    """

    def __init__(
        self,
        weight_label: str = "weight",
        processing_time_label: str = "processing_time",
        strict: bool = False,
    ):
        super().__init__(strict)
        self.weighted_label = weight_label
        self.processing_time = processing_time_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        weights: list[float] = self.get_data(tasks, jobs, self.weighted_label)
        processing_time: list[int] = self.get_data(tasks, jobs, self.processing_time)

        return [
            -proc_time / weight for weight, proc_time in zip(weights, processing_time)
        ]


class MinimumSlackTime(PriorityDispatchingRule):
    """
    Minimum Slack Time (MST) heuristic.

    This heuristic selects the job with the smallest slack time as the next job to be scheduled.
    """

    def __init__(
        self,
        due_date_label: str = "due_date",
        processing_time_label: str = "processing_time",
        release_time_label: str | None = None,
        strict: bool = False,
    ):
        super().__init__(strict)
        self.due_date_label = due_date_label
        self.processing_time_label = processing_time_label
        self.release_time_label = release_time_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        due_dates: list[int] = self.get_data(tasks, jobs, self.due_date_label)
        processing_time: list[int] = self.get_data(
            tasks, jobs, self.processing_time_label
        )

        if self.release_time_label is not None:
            release_dates: list[int] = self.get_data(
                tasks, jobs, self.release_time_label
            )

            return [
                max(time, release_date) + proc_time - due_date
                for due_date, proc_time, release_date in zip(
                    due_dates, processing_time, release_dates
                )
            ]

        return [
            time + proc_time - due_date
            for due_date, proc_time in zip(due_dates, processing_time)
        ]


class CriticalRatio(PriorityDispatchingRule):
    """
    Critical Ratio (CR) heuristic.

    This heuristic selects the job with the smallest critical ratio as the next job to be scheduled.
    """

    def __init__(
        self,
        due_date_label: str = "due_date",
        processing_time_label: str = "processing_time",
        release_time_label: str | None = None,
        strict: bool = False,
    ):
        super().__init__(strict)
        self.due_date_label = due_date_label
        self.processing_time_label = processing_time_label
        self.release_time_label = release_time_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        due_dates: list[int] = self.get_data(tasks, jobs, self.due_date_label)
        processing_time: list[int] = self.get_data(
            tasks, jobs, self.processing_time_label
        )

        if self.release_time_label is not None:
            release_dates: list[int] = self.get_data(
                tasks, jobs, self.release_time_label
            )

            return [
                (due_date - max(time, release_date)) / proc_time
                for due_date, proc_time, release_date in zip(
                    due_dates, processing_time, release_dates
                )
            ]

        return [
            (due_date - time) / proc_time
            for due_date, proc_time in zip(due_dates, processing_time)
        ]


class FirstInFirstOut(PriorityDispatchingRule):
    """
    First In First Out (FIFO) heuristic.

    This heuristic selects the job that has been in the waiting buffer the longest as the next job to be scheduled.
    """

    def __init__(self, release_time_label: str = "release_time", strict: bool = False):
        super().__init__(strict)
        self.release_time_label = release_time_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        release_dates: list[int] = self.get_data(tasks, jobs, self.release_time_label)

        return [time - release_date for release_date in release_dates]


class CostOverTime(PriorityDispatchingRule):
    """
    Cost OVER Time (CoverT) heuristic.
    This heuristic assign a cost value to each task
    """

    def __init__(
        self,
        weight_label: str = "weight",
        processing_time_label: str = "processing_time",
        due_date_label: str = "due_date",
        strict: bool = False,
    ):
        super().__init__(strict)
        self.weighted_label = weight_label
        self.processing_time = processing_time_label
        self.due_date_label = due_date_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        weights: list[float] = self.get_data(tasks, jobs, self.weighted_label)
        processing_time: list[int] = self.get_data(tasks, jobs, self.processing_time)
        due_dates: list[int] = self.get_data(tasks, jobs, self.due_date_label)

        P = sum(processing_time)

        return [
            (
                -proc_time / weight
                if due_date <= proc_time + time
                else (
                    -proc_time / weight * (P - time - proc_time) / (P - due_date)
                    if due_date < P
                    else 0
                )
            )
            for weight, proc_time, due_date in zip(weights, processing_time, due_dates)
        ]


class ApparentTardinessCost(PriorityDispatchingRule):
    """
    Modified Apparent Tardiness Cost (ATC) heuristic.
    This heuristic assign a cost value to each task
    """

    def __init__(
        self,
        lookahead: float,
        weight_label: str = "weight",
        processing_time_label: str = "processing_time",
        due_date_label: str = "due_date",
        strict: bool = False,
    ):
        super().__init__(strict)
        self.lookahead = lookahead

        self.weighted_label = weight_label
        self.processing_time = processing_time_label
        self.due_date_label = due_date_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        weights: list[float] = self.get_data(tasks, jobs, self.weighted_label)
        processing_time: list[int] = self.get_data(tasks, jobs, self.processing_time)
        due_dates: list[int] = self.get_data(tasks, jobs, self.due_date_label)

        P_mean = sum(processing_time) / len(processing_time)

        return [
            -proc_time
            / weight
            * exp(max(0, due_date - time - proc_time) / (self.lookahead * P_mean))
            for weight, proc_time, due_date in zip(weights, processing_time, due_dates)
        ]


class TrafficPriority(PriorityDispatchingRule):
    """
    Traffic Priority (TP) heuristic.
    This heuristic assign a cost value to each task
    """

    def __init__(
        self,
        K: float = 3.0,
        processing_time_label: str = "processing_time",
        due_date_label: str = "due_date",
        strict: bool = False,
    ):
        super().__init__(strict)
        self.K = K

        self.processing_time = processing_time_label
        self.due_date_label = due_date_label

    def priority_rule(
        self, tasks: dict[str, list[Any]], jobs: dict[str, list[Any]], time: int
    ) -> list[float]:
        processing_time: list[int] = self.get_data(tasks, jobs, self.processing_time)
        due_dates: list[int] = self.get_data(tasks, jobs, self.due_date_label)

        traffic_congestion_ratio = (
            len(processing_time) * sum(processing_time) / sum(due_dates)
        )

        weighted_edd = max(0, min(self.K / traffic_congestion_ratio - 0.5, 1))

        max_process_time = max(processing_time)
        max_due_date = max(due_dates)

        return [
            -(
                due_date / max_due_date * weighted_edd
                + proc_time / max_process_time * (1 - weighted_edd)
            )
            for proc_time, due_date in zip(processing_time, due_dates)
        ]

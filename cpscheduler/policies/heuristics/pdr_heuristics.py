from typing import Any, Sequence, TypeVar, TypeGuard
from numpy.typing import NDArray
from pandas import DataFrame

import numpy as np
import pandas as pd

from mypy_extensions import mypyc_attr

from ...utils import dataframe_to_structured

@mypyc_attr(allow_interpreted_subclasses=True)
class PriorityDispatchingRule:
    """
    Abstract class for Priority Dispatching Rule-based policies. To implement one, inherit from this class and implement
    the `priority_rule` method, addressing a priority value for each task. The tasks will be sorted in descending order
    of priority.
    """

    def priority_rule(self, obs: NDArray[np.void]) -> NDArray[np.float32]:
        """
        Implements a priority rule to sort the tasks in the waiting buffer by a given criterion.

        Parameters:
        - obs: NDArray[np.void] | DataFrame
            The current observation of the environment.
        """
        raise NotImplementedError


    def __call__(self, obs: NDArray[np.void] | DataFrame) -> NDArray[Any]:
        if isinstance(obs, DataFrame):
            obs = dataframe_to_structured(obs)

        priorities = self.priority_rule(obs)

        priority_queue = np.argsort(-priorities, stable=True).astype(np.int32)

        tasks_order: NDArray[Any] = np.take(obs['task_id'], priority_queue)

        return tasks_order


class ShortestProcessingTime(PriorityDispatchingRule):
    """
    Shortest Processing Time (SPT) heuristic.

    This heuristic selects the job with the shortest processing time as the next job to be scheduled.
    """
    def __init__(
            self,
            processing_time_label: str = 'processing_time'
        ):
        self.processing_time = processing_time_label


    def priority_rule(self, obs: NDArray[np.void]) -> NDArray[np.float32]:
        return -obs[self.processing_time]



class MostOperationsRemaining(PriorityDispatchingRule):
    """
    Most Operations Remaining (MOPNR) heuristic.

    This heuristic selects the earliest job to be done in the waiting buffer as the next job to be scheduled.
    """
    def __init__(
            self,
            job_label: str       = 'job',
            operation_label: str = 'operation'
        ):
        self.job = job_label
        self.operation = operation_label
    

    def priority_rule(self, obs: NDArray[np.void]) -> NDArray[np.float32]:
        priority_values = np.zeros(len(obs), dtype=np.float32)

        for job in np.unique(obs[self.job]):
            mask = obs[self.job] == job
            priority_values[mask] = obs[self.operation][mask].max() - obs[self.operation][mask]

        return priority_values



class MostWorkRemaining(PriorityDispatchingRule):
    """
    Most Work Remaining (MWKR) heuristic.

    This heuristic selects the job with the most work remaining as the next job to be scheduled.
    """
    def __init__(
            self,
            job_label: str       = 'job',
            operation_label: str = 'operation',
            processing_time_label: str = 'processing_time'
        ):
        self.job_label             = job_label
        self.operation_label       = operation_label
        self.processing_time_label = processing_time_label


    def priority_rule(self, obs: NDArray[np.void]) -> NDArray[np.float32]:
        priority_values = np.zeros(len(obs), dtype=np.float32)

        sorted_indices = np.argsort(obs, order=[self.job_label, self.operation_label])
        sorted_data    = obs[sorted_indices]

        for job in np.unique(obs[self.job_label]):
            mask = sorted_data[self.job_label] == job

            original_indices = sorted_indices[mask]
            processing_times = sorted_data[self.processing_time_label][mask]

            priority_values[original_indices] = np.cumsum(processing_times[::-1])[::-1]

        return priority_values


class EarliestDueDate(PriorityDispatchingRule):
    """
    Earliest Due Date (EDD) heuristic.

    This heuristic selects the job with the earliest due date as the next job to be scheduled.
    """
    def __init__(
            self,
            due_date_label: str = 'due_date'
        ):
        self.due_date_label = due_date_label


    def priority_rule(self, obs: NDArray[np.void]) -> NDArray[np.float32]:
        return -obs[self.due_date_label]


class WeightedShortestProcessingTime(PriorityDispatchingRule):
    """
    Weighted Shortest Processing Time (WSPT) heuristic.

    This heuristic selects the job with the shortest processing time as the next job to be scheduled, but the processing
    time is weighted by a given factor.
    """
    def __init__(
            self,
            weight: NDArray[np.floating[Any]],
            processing_time_label: str = 'processing_time',
            weighted_label: str = 'job',
        ):
        self.processing_time = processing_time_label
        self.weighted_label       = weighted_label

        self.weight = weight


    def priority_rule(self, obs: NDArray[np.void]) -> NDArray[np.float32]:
        priority_values: NDArray[np.float32] = (self.weight[obs[self.weighted_label]] / obs[self.processing_time]).astype(np.float32)

        return priority_values

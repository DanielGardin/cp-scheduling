from typing import Any, Sequence

from mypy_extensions import mypyc_attr

@mypyc_attr(allow_interpreted_subclasses=True)
class PriorityDispatchingRule:
    """
    Abstract class for Priority Dispatching Rule-based policies. To implement one, inherit from this class and implement
    the `priority_rule` method, addressing a priority value for each task. The tasks will be sorted in descending order
    of priority.
    """

    def priority_rule(self, obs: dict[str, list[Any]]) -> list[float]:
        """
        Implements a priority rule to sort the tasks in the waiting buffer by a given criterion.

        Parameters:
        - obs: dict[str, list[Any]]
            The current observation of the environment.
        """
        raise NotImplementedError


    def __call__(self, obs: dict[str, list[Any]]) -> list[int]:
        priority_values = self.priority_rule(obs)

        priorities = [(priority, i) for i, priority in enumerate(priority_values)]

        # Maximizing the priority values and breaking ties by selecting the task with the lowest index.
        tasks_order = [obs['task_id'][i] for _, i in sorted(priorities, key=lambda x: (-x[0], x[1]))]

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
        self.processing_time_label = processing_time_label


    def priority_rule(self, obs: dict[str, list[Any]]) -> list[float]:
        return [-processing_time for processing_time in obs[self.processing_time_label]]



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
        self.job_label      = job_label
        self.operation_label = operation_label
    

    def priority_rule(self, obs: dict[str, list[Any]]) -> list[float]:
        priority_values: list[float] = obs[self.operation_label].copy()

        max_ops: dict[Any, int] = {}
        for job, op in zip(obs[self.job_label], obs[self.operation_label]):
            if job not in max_ops:
                max_ops[job] = 0

            max_ops[job] = max(max_ops[job], op)
        
        for i, op in enumerate(priority_values):
            priority_values[i] = max_ops[obs[self.job_label][i]] - op

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


    def priority_rule(self, obs: dict[str, list[Any]]) -> list[float]:
        jobs             = obs[self.job_label]
        ops: list[int]   = obs[self.operation_label]
        procs: list[float] = obs[self.processing_time_label]


        cumulative_processing_times: dict[Any, list[float]] = {}

        for job, op, proc_time in zip(jobs, ops, procs):
            if job not in cumulative_processing_times:
                cumulative_processing_times[job] = []

            if len(cumulative_processing_times[job]) <= op:
                cumulative_processing_times[job].extend([0] * (op - len(cumulative_processing_times[job]) + 1))

            for i in range(op + 1):
                cumulative_processing_times[job][i] += proc_time

        priority_values = [
            cumulative_processing_times[job][op] for job, op in zip(jobs, ops)
        ]

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


    def priority_rule(self, obs: dict[str, list[Any]]) -> list[float]:
        return [-due_date for due_date in obs[self.due_date_label]]


class WeightedShortestProcessingTime(PriorityDispatchingRule):
    """
    Weighted Shortest Processing Time (WSPT) heuristic.

    This heuristic selects the job with the shortest processing time as the next job to be scheduled, but the processing
    time is weighted by a given factor.
    """
    def __init__(
            self,
            weight: list[float],
            processing_time_label: str = 'processing_time',
            weighted_label: str = 'job',
        ):
        self.processing_time = processing_time_label
        self.weighted_label  = weighted_label

        self.weight = weight


    def priority_rule(self, obs: dict[str, list[Any]]) -> list[float]:
        weight_ids: list[int]   = obs[self.weighted_label]
        proc_times: list[float] = obs[self.processing_time]

        priority_values = [
            self.weight[weight_id] / proc_time for weight_id, proc_time in zip(weight_ids, proc_times)
        ]

        return priority_values


class MinimumSlackTime(PriorityDispatchingRule):
    """
    Minimum Slack Time (MST) heuristic.

    This heuristic selects the job with the smallest slack time as the next job to be scheduled.
    """
    def __init__(
            self,
            due_date_label: str = 'due_date',
            processing_time_label: str = 'processing_time'
        ):
        self.due_date_label       = due_date_label
        self.processing_time_label = processing_time_label


    def priority_rule(self, obs: dict[str, list[Any]]) -> list[float]:
        due_dates: list[int]  = obs[self.due_date_label]
        proc_times: list[int] = obs[self.processing_time_label]
    
        priority_values = [
            float(due_date - proc_time) for due_date, proc_time in zip(due_dates, proc_times)
        ]

        return priority_values


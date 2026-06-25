"""Precedence rules for priority dispatching heuristics."""

from typing_extensions import override

from cpscheduler.environment.constants import TaskID, Time
from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.environment.utils import convert_to_list
from cpscheduler.heuristics.pdrs.base import StaticPriorityDispatchingRule


def _reverse_topological_sort(
    parents: dict[TaskID, list[TaskID]], n_tasks: int
) -> list[TaskID]:
    out_degree = [0] * n_tasks

    for parent_list in parents.values():
        for parent in parent_list:
            out_degree[parent] += 1

    queue = [task_id for task_id, deg in enumerate(out_degree) if deg == 0]
    idx = 0

    while idx < len(queue):
        task_id = queue[idx]
        idx += 1

        if task_id not in parents:
            continue

        for parent in parents[task_id]:
            out_degree[parent] -= 1

            if out_degree[parent] == 0:
                queue.append(parent)

    return queue


class MostWorkRemaining(StaticPriorityDispatchingRule):
    """Most Work Remaining (MWKR) heuristic.

    Prioritizes tasks based on the total processing time of the task and all
    its successors.
    """

    processing_time: str
    precedence_label: str

    def __init__(
        self,
        processing_time: str = "processing_time",
        precedence_label: str = "precedence",
        seed: int | None = None,
    ) -> None:
        """Initialize the Most Work Remaining heuristic.

        Parameters
        ----------
        processing_time : str, optional
            The label for the processing time attribute in the observation.
            Default is "processing_time".

        precedence_label : str, optional
            The label for the precedence relationship in the observation.
            Default is "precedence".

        seed : int | None, optional
            The seed for the random number generator. Default is None.

        """
        super().__init__(seed)

        self.processing_time = processing_time
        self.precedence_label = precedence_label

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        parents: dict[TaskID, list[TaskID]] = obs.global_state[
            self.precedence_label
        ]
        processing_times: list[Time] = obs.task[self.processing_time]

        priorities = processing_times.copy()

        for task_id in _reverse_topological_sort(parents, obs.n_tasks):
            if task_id not in parents:
                continue

            prio = priorities[task_id]

            for parent in parents[task_id]:
                priorities[parent] += prio

        return convert_to_list(priorities, float)


class MostOperationsRemaining(StaticPriorityDispatchingRule):
    """
    Most Operations Remaining (MOPNR) heuristic.

    Prioritizes tasks based on the number of remaining future operations.
    """

    precedence_label: str

    def __init__(
        self, precedence_label: str = "precedence", seed: int | None = None
    ) -> None:
        """Initialize the Most Operations Remaining heuristic.

        Parameters
        ----------
        precedence_label : str, optional
            The label for the precedence relationship in the observation.
            Default is "precedence".

        seed : int | None, optional
            The seed for the random number generator. Default is None.

        """
        super().__init__(seed)

        self.precedence_label = precedence_label

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        parents: dict[TaskID, list[TaskID]] = obs.global_state[
            self.precedence_label
        ]

        priorities = [1] * obs.n_tasks

        for task_id in _reverse_topological_sort(parents, obs.n_tasks):
            if task_id not in parents:
                continue

            prio = priorities[task_id]

            for parent in parents[task_id]:
                priorities[parent] += prio

        return convert_to_list(priorities, float)

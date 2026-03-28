from typing import Any
from collections.abc import Iterable, Mapping, Sequence
from typing_extensions import Self

from cpscheduler.environment.constants import TaskID, Int
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.utils import convert_to_list


def topological_sort(
    precedence_map: dict[TaskID, list[TaskID]], n_tasks: int
) -> list[TaskID]:
    """
    Perform a topological sort on a directed acyclic graph.

    Parameters
    ----------
    precedence_map: dict
        A dictionary containing the precedence relationships between the tasks.

    in_degree: list
        A list containing the in-degree of each task.

    Returns
    -------
    list
        A list containing the tasks in topological order
    """
    in_degree = [0] * n_tasks
    for children in precedence_map.values():
        for child in children:
            in_degree[child] += 1

    queue = [task for task, degree in enumerate(in_degree) if degree == 0]

    topological_order: list[int] = []

    idx = 0
    while idx < len(queue):
        vertex = queue[idx]
        idx += 1

        if vertex not in precedence_map or not precedence_map[vertex]:
            continue

        topological_order.append(vertex)

        for child in precedence_map[vertex]:
            in_degree[child] -= 1

            if in_degree[child] == 0:
                queue.append(child)

    return topological_order


class PrecedenceConstraint(Constraint):
    """
    Precedence constraint for the scheduling environment.
    This constraint defines the precedence relationships between tasks, where some tasks
    must be completed before others can start.

    Arguments:
        precedence: Mapping[int, Iterable[int]]
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 must be completed before task 2 can start,
            the precedence mapping would be {2: [1]}.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    parents: dict[TaskID, list[TaskID]]
    "A mapping of task IDs to their parent task IDs."

    children: dict[TaskID, list[TaskID]]
    "A mapping of task IDs to their child task IDs."

    def __init__(self, precedence: Mapping[Int, Sequence[Int]]):
        self.parents = {
            TaskID(child_id): convert_to_list(parent_ids, TaskID)
            for child_id, parent_ids in precedence.items()
        }

        self.children = {}
        for child_id, parent_ids in self.parents.items():
            for parent_id in parent_ids:
                if parent_id not in self.children:
                    self.children[parent_id] = []

                self.children[parent_id].append(child_id)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.parents,),
            (),
        )

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[Int, Int]]) -> Self:
        """
        Create a PrecedenceConstraint from a list of edges.

        Arguments:
            edges: Iterable[tuple[int, int]]
                A list of tuples representing the edges of the precedence graph.
                Each tuple (parent, child) indicates that the parent task must be completed
                before the child task can start.
        """
        precedence: dict[Int, list[Int]] = {}

        for parent, child in edges:
            if parent not in precedence:
                precedence[parent] = []

            precedence[parent].append(child)

        return cls(precedence)

    def is_intree(self) -> bool:
        "Check if the precedence graph is an in-tree."
        for parents in self.parents.values():
            if len(parents) > 1:
                return False

        return True

    def is_outtree(self) -> bool:
        "Check if the precedence graph is an out-tree."
        for children in self.children.values():
            if len(children) > 1:
                return False

        return True

    def reset(self, state: ScheduleState) -> None:
        for task_id in topological_sort(self.children, state.n_tasks):
            end_time = state.get_end_lb(task_id)

            for child_id in self.children[task_id]:
                state.tight_start_lb(child_id, end_time)

    def on_start_lb(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        if task_id in self.children:
            end_time = state.get_end_lb(task_id)

            for child_id in self.children[task_id]:
                state.tight_start_lb(child_id, end_time)

    def on_start_ub(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        if task_id in self.parents:
            start_time = state.get_start_ub(task_id)

            for parent_id in self.parents[task_id]:
                state.tight_end_ub(parent_id, start_time)

    def on_end_lb(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        self.on_start_lb(task_id, machine_id, state)

    def on_end_ub(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        self.on_start_ub(task_id, machine_id, state)


    def get_entry(self) -> str:
        intree = self.is_intree()
        outtree = self.is_outtree()

        graph = "prec"
        if intree and outtree:
            graph = "chains"

        elif intree:
            graph = "intree"

        elif outtree:
            graph = "outtree"

        return graph


class NoWaitConstraint(PrecedenceConstraint):
    """

    No-wait precedence constraint for the scheduling environment.

    This constraint is a specialized version of the PrecedenceConstraint that enforces
    that tasks must be executed back-to-back without any waiting time in between.

    Arguments:
        precedence: Mapping[int, Iterable[int]]
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 must be completed before task 2 can start,
            the precedence mapping would be {2: [1]}.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    def __init__(self, precedence: Mapping[Int, Sequence[Int]]):
        super().__init__(precedence)

        if not self.is_intree():
            raise ValueError("No-wait constraint must be an in-tree.")

    def on_assignment(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        if task_id in self.children:
            end_time = state.get_end_lb(task_id)

            for child_id in self.children[task_id]:
                state.tight_start_ub(child_id, end_time)

    def on_start_lb(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        super().on_start_lb(task_id, machine_id, state)

        if task_id in self.parents:
            start_time = state.get_start_lb(task_id)

            for parent_id in self.parents[task_id]:
                state.tight_end_lb(parent_id, start_time)

    def get_entry(self) -> str:
        return "nwt"


class ORPrecedenceConstraint(PrecedenceConstraint):
    """
    OR precedence constraint for the scheduling environment.

    This constraint defines a precedence relationship where at least one of the
    parent tasks must be completed before the child task can start.

    Arguments:
        precedence: Mapping[int, Iterable[int]]
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 or task 2 must be completed before task 3 can start,
            the precedence mapping would be {3: [1, 2]}.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    def __init__(self, precedence: Mapping[Int, Sequence[Int]]):
        super().__init__(precedence)

    def reset(self, state: ScheduleState) -> None:
        for task_id in topological_sort(self.children, state.n_tasks):
            if task_id in self.parents:
                earliest_start = min(
                    state.get_end_lb(parent_id)
                    for parent_id in self.parents[task_id]
                )
                state.tight_start_lb(task_id, earliest_start)

    def on_start_lb(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        if task_id in self.children:
            for child_id in self.children[task_id]:
                earliest_start = min(
                    state.get_end_lb(parent_id)
                    for parent_id in self.parents[child_id]
                )
                state.tight_start_lb(child_id, earliest_start)

    def on_start_ub(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        if task_id in self.parents:
            start_time = state.get_start_ub(task_id)

            feasible_parents = [
                parent_id
                for parent_id in self.parents[task_id]
                if state.get_end_ub(parent_id) <= start_time
            ]

            if not feasible_parents:
                any_parent_id = self.parents[task_id][0]
                state.tight_end_ub(any_parent_id, start_time)

            elif len(feasible_parents) == 1:
                state.tight_end_ub(feasible_parents[0], start_time)

            # When there are multiple feasible parents, we cannot decide which
            # one to tighten, so we do nothing.

    def get_entry(self) -> str:
        return "or-prec"

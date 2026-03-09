from typing import Any
from collections.abc import Iterable, Mapping, Sequence
from typing_extensions import Self

from cpscheduler.utils.general_algo import topological_sort

from cpscheduler.environment.constants import TaskID, Int
from cpscheduler.environment.state.events import Event
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint


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

    precedence: dict[TaskID, list[TaskID]]
    transposed_precedence: dict[TaskID, list[TaskID]]

    def __init__(self, precedence: Mapping[Int, Sequence[Int]]):
        self.precedence = {
            TaskID(task): [TaskID(child) for child in children]
            for task, children in precedence.items()
        }

        self.transposed_precedence = {}

        for task_id, children in self.precedence.items():
            for child_id in children:
                self.transposed_precedence.setdefault(child_id, []).append(
                    task_id
                )

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.precedence,),
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
        for tasks in self.precedence.values():
            if len(tasks) > 1:
                return False

        return True

    def is_outtree(self) -> bool:
        "Check if the precedence graph is an out-tree."
        n_children = 0
        unique_children: set[TaskID] = set()

        for tasks in self.precedence.values():
            n_children += len(tasks)
            unique_children.update(tasks)

        return n_children == len(unique_children)

    def reset(self, state: ScheduleState) -> None:
        for task_id in topological_sort(self.precedence, state.n_tasks):
            end_time = state.get_end_lb(task_id)

            for child_id in self.precedence[task_id]:
                state.tight_start_lb(child_id, end_time)

    def propagate(self, event: Event, state: ScheduleState) -> None:
        task_id = event.task_id

        if event.is_lower_bound() and task_id in self.precedence:
            end_time = state.get_end_lb(task_id)

            for child_id in self.precedence[task_id]:
                state.tight_start_lb(child_id, end_time)

        elif event.is_upper_bound() and task_id in self.transposed_precedence:
            start_time = state.get_start_ub(task_id)

            for parent_id in self.transposed_precedence[task_id]:
                state.tight_end_ub(parent_id, start_time)

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

    def propagate(self, event: Event, state: ScheduleState) -> None:
        super().propagate(event, state)

        task_id = event.task_id

        if state.is_fixed(task_id):
            if task_id in self.precedence:
                end_time = state.get_end_lb(task_id)

                for child_id in self.precedence[task_id]:
                    state.tight_start_ub(child_id, end_time)

        elif event.is_lower_bound() and task_id in self.transposed_precedence:
            start_time = state.get_start_lb(task_id)

            for parent_id in self.transposed_precedence[task_id]:
                state.tight_end_lb(parent_id, start_time)

    def get_entry(self) -> str:
        return "nwt"

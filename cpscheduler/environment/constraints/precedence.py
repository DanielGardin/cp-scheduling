"""Precedence constraints for the scheduling environment."""

from collections.abc import Iterable, Mapping, Sequence
from typing import override

from typing_extensions import Self

from cpscheduler.environment.constants import Int, MachineID, TaskID
from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.instance import (
    UNSET,
    GlobalFeature,
    ProblemInstance,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


def topological_sort(
    precedence_map: dict[TaskID, list[TaskID]],
    n_tasks: int,
    remove_leaves: bool = True,
) -> list[TaskID]:
    """Perform a topological sort on a directed acyclic graph.

    Parameters
    ----------
    precedence_map: dict[TaskID, list[TaskID]]
        A mapping of task IDs to a list of their child task IDs, representing the
        precedence relationships between tasks.

    n_tasks: int
        The total number of tasks in the graph.

    remove_leaves: bool, optional
        If True, only return the internal nodes of the graph (i.e., tasks that have
        children). If False, return all tasks in topological order. Default is True.

    Returns
    -------
    list[TaskID]
        A list containing the tasks in topological order

    """
    in_degree = [0] * n_tasks
    for children in precedence_map.values():
        for child in children:
            in_degree[child] += 1

    queue = [task for task, degree in enumerate(in_degree) if degree == 0]

    topological_order: list[TaskID] = []

    idx = 0
    while idx < len(queue):
        vertex = queue[idx]
        idx += 1

        if not remove_leaves or (precedence_map.get(vertex)):
            topological_order.append(vertex)

            for child in precedence_map.get(vertex, []):
                in_degree[child] -= 1

                if in_degree[child] == 0:
                    queue.append(child)

    return topological_order


def _inverse_graph(
    graph: dict[TaskID, list[TaskID]],
) -> dict[TaskID, list[TaskID]]:
    inverse: dict[TaskID, list[TaskID]] = {}
    for child_id, parent_ids in graph.items():
        for parent_id in parent_ids:
            inverse.setdefault(parent_id, []).append(child_id)

    return inverse


class PrecedenceConstraint(Constraint):
    """Precedence constraint for the scheduling environment.

    This constraint defines the precedence relationships between tasks, where some tasks
    must be completed before others can start.
    """

    parents: GlobalFeature[dict[TaskID, list[TaskID]]]
    "A mapping of task IDs to their parent task IDs."

    children: dict[TaskID, list[TaskID]]
    "A mapping of task IDs to their child task IDs."

    def __init__(
        self,
        precedence: Mapping[Int, Sequence[Int]] | None = None,
        name: str = "precedence",
    ):
        """Initialize the Precedence Constraint.

        Parameters
        ----------
        precedence: Mapping[int, Iterable[int]] | None
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 must be completed before task
            2 can start, the precedence mapping would be {2: [1]}.

        name: str, optional
            An optional name for the adjacency feature.

        """
        self.parents = GlobalFeature(
            name=name,
            semantic="adjacency",
            default=(
                {
                    TaskID(child_id): convert_to_list(parent_ids, TaskID)
                    for child_id, parent_ids in precedence.items()
                }
                if precedence is not None
                else UNSET
            ),
        )

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[Int, Int]], name: str = "precedence"
    ) -> Self:
        """Create a PrecedenceConstraint from a list of edges.

        Parameters
        ----------
        edges: Iterable[tuple[int, int]]
            A list of tuples representing the edges of the precedence graph.
            Each tuple (parent, child) indicates that the parent task must be completed
            before the child task can start.

        name: str, optional
            An optional name for the adjacency feature.

        """
        precedence: dict[Int, list[Int]] = {}

        for parent, child in edges:
            if child not in precedence:
                precedence[child] = []

            precedence[child].append(parent)

        return cls(precedence, name)

    def add_precedence(self, parent_id: Int, child_id: Int) -> None:
        """Add a precedence relationship between two tasks."""
        if not self.parents.loaded:
            self.parents.own_data({})

        parent, child = TaskID(parent_id), TaskID(child_id)

        self.parents.value.setdefault(child, []).append(parent)

    def remove_precedence(self, parent_id: Int, child_id: Int) -> None:
        """Remove a precedence relationship between two tasks."""
        if not self.parents.loaded:
            raise ValueError(
                "Cannot remove precedence, no precedence graph has been loaded."
            )

        parent, child = TaskID(parent_id), TaskID(child_id)

        children = self.parents.value[child]
        children.remove(parent)

        if not children:
            del self.parents.value[child]

    def add_chain(self, chain: Sequence[Int]) -> None:
        """Add a chain of precedence relationships between a sequence of tasks."""
        if not self.parents.loaded:
            self.parents.own_data({})

        if not chain:
            raise ValueError("Invalid empty chain in precedence.")

        parents = self.parents.value

        parent = TaskID(chain[0])

        for i in range(1, len(chain)):
            child = TaskID(chain[i])
            parents.setdefault(child, []).append(parent)

            parent = child

    def is_intree(self) -> bool:
        """Check if the precedence graph is an in-tree."""
        if self.parents.loaded:
            for parents in self.parents.value.values():
                if len(parents) > 1:
                    return False

            return True

        raise ValueError("is_intree: Precedence graph has not been loaded yet.")

    def is_outtree(self) -> bool:
        """Check if the precedence graph is an out-tree."""
        if self.parents.loaded:
            for children in _inverse_graph(self.parents.value).values():
                if len(children) > 1:
                    return False

            return True

        raise ValueError(
            "is_outtree: Precedence graph has not been loaded yet."
        )

    @override
    def get_features(self) -> list[GlobalFeature]:
        return [self.parents]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        self.children = _inverse_graph(self.parents.value)

    @override
    def reset(self, state: ScheduleState) -> None:
        for task_id in topological_sort(self.children, state.n_tasks):
            end_time = state.get_end_lb(task_id)

            for child_id in self.children[task_id]:
                state.tight_start_lb(child_id, end_time)
                state.add_dependency(child_id, f"precedence:{task_id}")

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if task_id in self.children:
            for child_id in self.children[task_id]:
                state.resolve_dependency(child_id, f"precedence:{task_id}")

    @override
    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if task_id in self.children:
            for child_id in self.children[task_id]:
                state.add_dependency(child_id, f"precedence:{task_id}")

    @override
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if task_id in self.children:
            end_time = state.get_end_lb(task_id)

            for child_id in self.children[task_id]:
                state.tight_start_lb(child_id, end_time)

    @override
    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        parents = self.parents.value

        if task_id in parents:
            start_time = state.get_start_ub(task_id)

            for parent_id in parents[task_id]:
                state.tight_end_ub(parent_id, start_time)

    @override
    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.on_start_lb(task_id, machine_id, state)

    @override
    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.on_start_ub(task_id, machine_id, state)

    @override
    def get_entry(self) -> str:
        if not self.parents.loaded:
            return self.get_general_entry()

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

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "prec"


class NoWaitConstraint(PrecedenceConstraint):
    """No-wait precedence constraint for the scheduling environment.

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

    def __init__(
        self,
        precedence: Mapping[Int, Sequence[Int]] | None = None,
        name: str = "no_wait_precedence",
    ):
        """Initialize the No-Wait Constraint.

        Parameters
        ----------
        precedence: Mapping[int, Iterable[int]] | None
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 must be completed before task
            2 can start, the precedence mapping would be {2: [1]}.

        name: str, optional
            An optional name for the adjacency feature.

        """
        super().__init__(precedence, name)

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[Int, Int]], name: str = "no_wait_precedence"
    ) -> Self:
        """Create a No-Wait Precedence Constraint from a list of edges.

        Parameters
        ----------
        edges: Iterable[tuple[int, int]]
            A list of tuples representing the edges of the precedence graph.
            Each tuple (parent, child) indicates that the parent task must be completed
            before the child task can start.

        name: str, optional
            An optional name for the adjacency feature.

        """
        return cls.from_edges(edges, name)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)

        if not self.is_intree():
            raise ValueError("No-wait constraint must be an in-tree.")

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if task_id in self.children:
            end_time = state.get_end_lb(task_id)

            for child_id in self.children[task_id]:
                state.tight_start_ub(child_id, end_time)

    @override
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        super().on_start_lb(task_id, machine_id, state)

        parents = self.parents.value

        if task_id in parents:
            start_time = state.get_start_lb(task_id)

            for parent_id in parents[task_id]:
                state.tight_end_lb(parent_id, start_time)

    @override
    def get_entry(self) -> str:
        return f"{super().get_entry()}, nwt"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "prec, nwt"


class ORPrecedenceConstraint(PrecedenceConstraint):
    """OR precedence constraint for the scheduling environment.

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

    def __init__(
        self,
        precedence: Mapping[Int, Sequence[Int]] | None = None,
        name: str = "or_precedence",
    ):
        """Initialize the OR Precedence Constraint.

        Parameters
        ----------
        precedence: Mapping[int, Iterable[int]] | None
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 must be completed before task
            2 can start, the precedence mapping would be {2: [1]}.

        name: str, optional
            An optional name for the adjacency feature.

        """
        super().__init__(precedence, name)

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[Int, Int]], name: str = "or_precedence"
    ) -> Self:
        """Create a OR Precedence Constraint from a list of edges.

        Parameters
        ----------
        edges: Iterable[tuple[int, int]]
            A list of tuples representing the edges of the precedence graph.
            Each tuple (parent, child) indicates that the parent task must be completed
            before the child task can start.

        name: str, optional
            An optional name for the adjacency feature.

        """
        return cls.from_edges(edges, name)

    @override
    def reset(self, state: ScheduleState) -> None:
        tasks = topological_sort(self.children, state.n_tasks, False)

        parents = self.parents.value

        for task_id in tasks:
            if task_id in parents:
                earliest_start = min(
                    state.get_end_lb(parent_id)
                    for parent_id in parents[task_id]
                )
                state.tight_start_lb(task_id, earliest_start)

    @override
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        parents = self.parents.value

        if task_id in self.children:
            for child_id in self.children[task_id]:
                earliest_start = min(
                    state.get_end_lb(parent_id)
                    for parent_id in parents[child_id]
                )
                state.tight_start_lb(child_id, earliest_start)

    @override
    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        parents = self.parents.value

        if task_id in parents:
            start_time = state.get_start_ub(task_id)

            feasible_parents = [
                parent_id
                for parent_id in parents[task_id]
                if state.get_end_ub(parent_id) <= start_time
            ]

            if not feasible_parents:
                any_parent_id = parents[task_id][0]
                state.tight_end_ub(any_parent_id, start_time)

            elif len(feasible_parents) == 1:
                state.tight_end_ub(feasible_parents[0], start_time)

            # When there are multiple feasible parents, we cannot decide which
            # one to tighten, so we do nothing.

    @override
    def get_entry(self) -> str:
        return f"or-{super().get_entry()}"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "or-prec"

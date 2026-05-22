from collections.abc import Iterable

from cpscheduler.environment.constants import TaskID

from cpscheduler.environment.constraints.base import Constraint


class DisjunctiveConstraint(Constraint):
    """Event-driven disjunctive constraint over an arbitrary graph.

    For every edge (i, j) in the disjunctive graph, ensure that
        (S_i >= C_j) OR (S_j >= C_i)

    Attributes:
        disjunctive_tag: Feature to be imported/exported to the task instance.
        edges: Iterable containing pairs of disjunctive tasks (i, j).
    """

    adjacency: dict[TaskID, set[TaskID]]

    def __init__(
        self,
        disjunctive_tag: str = "disjunctive_tasks",
        edges: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        self.adjacency = {}

        if edges is not None:
            for u, v in edges:
                i, j = TaskID(u), TaskID(v)

                if i == j:
                    continue

                self.adjacency.setdefault(i, set()).add(j)
                self.adjacency.setdefault(j, set()).add(i)

        raise NotImplementedError()

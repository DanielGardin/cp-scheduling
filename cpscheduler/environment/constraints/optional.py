"""Constraints related to optional tasks."""

from collections.abc import Iterable

from typing_extensions import override

import cpscheduler.environment.utils.debug as debug
from cpscheduler.environment.constants import Int, TaskID
from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


# TODO: Convert external information as Features
class AtMostOneConstraint(Constraint):
    """Alternative constraint where at most one task have to be processed in each one of the groups."""

    task_groups: list[list[TaskID]]

    current_tasks: list[set[TaskID]]

    def __init__(self, task_groups: Iterable[Iterable[Int]]) -> None:
        """Initialize the AtMostOneConstraint.

        Parameters
        ----------
        task_groups: Iterable[Iterable[Int]]
            An iterable of iterables, where each inner iterable represents a group of task IDs.

        """
        self.task_groups = [
            convert_to_list(tasks, TaskID) for tasks in task_groups
        ]

        self.current_tasks = []

    def add_group(self, task_group: Iterable[Int]) -> None:
        """Add a new group of tasks to the constraint."""
        self.task_groups.append(convert_to_list(task_group, TaskID))

    def remove_group(self, group_id: int) -> None:
        """Remove a group of tasks from the constraint by its index."""
        if 0 <= group_id < len(self.task_groups):
            del self.task_groups[group_id]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for tasks in self.task_groups:
            for task in tasks:
                debug.task_bounds(task, instance, type(self).__name__)

    @override
    def reset(self, state: ScheduleState) -> None:
        self.current_tasks = [set(tasks) for tasks in self.task_groups]
        all_tasks = {task for tasks in self.task_groups for task in tasks}

        # Tasks decided from the start emit no presence or absence events
        for task_id in sorted(all_tasks):
            if state.is_present(task_id):
                self.on_presence(task_id, state)

    @override
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            if task_id not in tasks:
                continue

            tasks.remove(task_id)

            for other_task in tasks:
                state.forbid_task(other_task)

            tasks.clear()

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            tasks.discard(task_id)


class ExactlyOneConstraint(AtMostOneConstraint):
    """Alternative constraint where exactly one task have to be processed in each one of the groups."""

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        if any(not tasks for tasks in self.task_groups):
            raise ValueError(
                "Cannot enforce exactly one task in an empty set of tasks."
            )

        super().initialize(instance)

    @override
    def reset(self, state: ScheduleState) -> None:
        super().reset(state)

        for tasks in self.current_tasks:
            if len(tasks) == 1:
                remaining_task = next(iter(tasks))
                tasks.clear()
                state.require_task(remaining_task)

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            if task_id not in tasks:
                continue

            tasks.remove(task_id)

            if len(tasks) == 1:
                remaining_task = next(iter(tasks))
                tasks.clear()
                state.require_task(remaining_task)

            elif not tasks:
                state.fail(task_id)


# TODO: Convert external information as Features
class AllOrNothingConstraint(Constraint):
    """Alternative constraint where the tasks of each group are all processed or all rejected.

    A task becoming present requires the rest of its group, and a task becoming
    absent forbids the rest of its group.

    Groups sharing a task are merged into components, as their tasks must all
    agree. The first decision in a component settles all its tasks and removes
    the component, so each one is propagated once, in O(tasks in the component).
    """

    backtrack_safe = True

    task_groups: list[list[TaskID]]

    _components: list[list[TaskID]]
    _component_of: dict[TaskID, int]

    _settled: list[int]
    "The components settled so far, in order."
    _marks: list[int]
    "Per checkpoint, how many components were settled."

    def __init__(self, task_groups: Iterable[Iterable[Int]] = ()) -> None:
        """Initialize the AllOrNothingConstraint.

        Parameters
        ----------
        task_groups: Iterable[Iterable[Int]]
            An iterable of iterables, where each inner iterable represents a group of task IDs.

        """
        self.task_groups = [
            convert_to_list(tasks, TaskID) for tasks in task_groups
        ]

        self._components = []
        self._component_of = {}
        self._settled = []
        self._marks = []

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        # Each group links its tasks to its first one, which is enough to
        # connect them, so components are only searched from first tasks.
        neighbors: dict[TaskID, list[TaskID]] = {}

        for tasks in self.task_groups:
            for task in tasks:
                debug.task_bounds(task, instance, type(self).__name__)
                neighbors.setdefault(task, [])

            for task in tasks[1:]:
                neighbors[tasks[0]].append(task)
                neighbors[task].append(tasks[0])

        visited: list[bool] = [False] * instance.n_tasks
        self._components = []

        for tasks in self.task_groups:
            if not tasks or visited[tasks[0]]:
                continue

            root = tasks[0]

            visited[root] = True
            stack = [root]
            component: list[TaskID] = []

            while stack:
                task = stack.pop()
                component.append(task)

                for neighbor in neighbors[task]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            if len(component) > 1:
                self._components.append(component)

    @override
    def reset(self, state: ScheduleState) -> None:
        self._component_of = {
            task: component_id
            for component_id, tasks in enumerate(self._components)
            for task in tasks
        }
        self._settled = []
        self._marks = []

        # Tasks decided from the start emit no presence or absence events
        for task_id in sorted(self._component_of):
            if state.is_present(task_id):
                self.on_presence(task_id, state)

            elif state.is_absent(task_id):
                self.on_absence(task_id, state)

    @override
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        component_id = self._component_of.get(task_id)
        if component_id is None:
            return

        for task in self._components[component_id]:
            del self._component_of[task]
            state.require_task(task)

        self._settled.append(component_id)

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        component_id = self._component_of.get(task_id)
        if component_id is None:
            return

        for task in self._components[component_id]:
            del self._component_of[task]
            state.forbid_task(task)

        self._settled.append(component_id)

    @override
    def checkpoint(self, mark: int) -> None:
        del self._marks[mark:]
        self._marks.append(len(self._settled))

    @override
    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        n_settled = self._marks[mark]
        del self._marks[mark:]

        while len(self._settled) > n_settled:
            component_id = self._settled.pop()

            for task in self._components[component_id]:
                self._component_of[task] = component_id


class RejectableConstraint(AllOrNothingConstraint):
    """Constraint that allows whole jobs to be rejected.

    Every task becomes optional, and the tasks of a job are either all
    processed or all rejected, as in scheduling with rejection, where a
    rejected job usually incurs a penalty (see `RejectionCost`).
    Tasks that may be left out independently of their job are modeled by
    `OptionalityConstraint` instead.

    This is made explicit by the "rej" entry, instead of "opt".
    """

    def __init__(self) -> None:
        """Initialize the RejectableConstraint."""
        super().__init__()

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id in range(instance.n_tasks):
            instance.set_optionality(task_id)

        self.task_groups = [list(tasks) for tasks in instance.job_tasks]

        super().initialize(instance)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "rej"

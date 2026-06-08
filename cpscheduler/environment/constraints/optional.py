"""Constraints related to optional tasks."""

from collections.abc import Iterable
from typing import override

import cpscheduler.environment.utils.debug as debug
from cpscheduler.environment.constants import Int, MachineID, TaskID
from cpscheduler.environment.constraints.base import (
    Constraint,
    PassiveConstraint,
)
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class RejectableConstraint(PassiveConstraint):
    """Semantic constraint that allows tasks to be rejected.

    This constraint allows to model scheduling problems where optionality has
    the semantics of rejection.
    The difference between optional and rejectable tasks, in this context, is
    that usually optional tasks are not considered in the objective function,
    while rejectable tasks are explicitly allowed to be rejected, incurring a
    penalty cost.

    This is made explicit by the "rej" entry, instead of "opt".
    """

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id in range(instance.n_tasks):
            instance.set_optionality(task_id)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "rej"


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
            if not tasks:
                raise ValueError(
                    "Cannot enforce exactly one constraint in a empty set of tasks."
                )

            for task in tasks:
                debug.task_bounds(task, instance, type(self).__name__)

    @override
    def reset(self, state: ScheduleState) -> None:
        self.current_tasks = [set(tasks) for tasks in self.task_groups]

    @override
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            if task_id not in tasks:
                continue

            tasks.remove(task_id)

            for other_task in tasks:
                state.forbid_task(other_task)

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.on_presence(task_id, state)

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            tasks.discard(task_id)


# TODO: Convert external information as Features
class ExactlyOneConstraint(AtMostOneConstraint):
    """Alternative constraint where exactly one task have to be processed in each one of the groups."""

    @override
    def reset(self, state: ScheduleState) -> None:
        super().reset(state)

        for tasks in self.current_tasks:
            if len(tasks) == 1:
                remaining_task = next(iter(tasks))
                state.require_task(remaining_task)
                tasks.remove(remaining_task)

    @override
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            if task_id not in tasks:
                continue

            tasks.remove(task_id)

            for other_task in tasks:
                state.forbid_task(other_task)

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.on_presence(task_id, state)

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            tasks.discard(task_id)

            if len(tasks) == 1:
                remaining_task = next(iter(tasks))
                state.require_task(remaining_task)
                tasks.remove(remaining_task)

            elif len(tasks) == 0:
                state.fail()

"""Disjunctive constraints for the scheduling environment.

Disjunctive constraints limit the execution of tasks such that certain tasks cannot
overlap in time.
"""

from collections.abc import Iterable
from typing import override

import cpscheduler.environment.utils.debug as debug
from cpscheduler.environment.constants import Int, MachineID, TaskID
from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.instance import (
    UNSET,
    GlobalFeature,
    ProblemInstance,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list, extend_list


class NonOverlapConstraint(Constraint):
    """Constraint for non-overlapping tasks at any time.

    This constraint ensures that tasks within the same group cannot overlap,
    independently of their assignments.
    """

    groups: GlobalFeature[list[list[TaskID]]]

    current_groups: list[set[TaskID]]

    def __init__(
        self,
        groups_tag: str = "non_overlap_groups",
        task_groups: Iterable[Iterable[Int]] | None = None,
    ):
        """Initialize the Non Overlap Constraint.

        Parameters
        ----------
        groups_tag : str, optional
            The tag for the groups feature.

        task_groups : Iterable[Iterable[Int]], optional
            An optional iterable of task groups, where each group is an iterable of task IDs.

        """
        self.groups = GlobalFeature(
            name=groups_tag,
            semantic="task",
            default=(
                [
                    convert_to_list(task_group, TaskID)
                    for task_group in task_groups
                ]
                if task_groups is not None
                else UNSET
            ),
        )

    def add_task(self, group_id: Int, task: Int) -> None:
        """Add a task to a specific group."""
        if not self.groups.loaded:
            self.groups.own_data([])

        group = int(group_id)

        extend_list(self.groups.value, group + 1, list)

        self.groups.value[group].append(TaskID(task))

    def add_group(self, task_group: Iterable[Int]) -> None:
        """Add a new group of tasks."""
        if not self.groups.loaded:
            self.groups.own_data([])

        self.groups.value.append(convert_to_list(task_group, TaskID))

    def remove_group(self, group_id: Int) -> None:
        """Remove a group of tasks by its ID."""
        self.groups.value[int(group_id)].clear()

    @override
    def get_features(self) -> list[GlobalFeature]:
        return [self.groups]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        if instance.debug:
            for tasks in self.groups.value:
                for task in tasks:
                    debug.task_bounds(task, instance, "NonOverlapConstraint")

    @override
    def reset(self, state: ScheduleState) -> None:
        self.current_groups = [set(group) for group in self.groups.value]

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for group_tasks in self.current_groups:
            if task_id not in group_tasks:
                continue

            group_tasks.remove(task_id)

            end_time = state.get_end_lb(task_id)

            for other_task_id in group_tasks:
                state.tight_start_lb(other_task_id, end_time)

    @override
    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for i, group_tasks in enumerate(self.groups.value):
            if task_id not in group_tasks:
                continue

            cur_group_tasks = self.current_groups[i]
            for other_task_id in cur_group_tasks:
                state.reset_bounds(other_task_id)

            self.current_groups[i].add(task_id)

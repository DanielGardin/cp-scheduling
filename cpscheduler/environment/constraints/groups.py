from collections.abc import Iterable

from cpscheduler.environment.constants import TaskID, MachineID, Int
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint

import cpscheduler.environment.debug as debug

class NonOverlapConstraint(Constraint):
    __slots__ = ("groups_map", "current_groups")

    groups_map: list[set[TaskID]]

    current_groups: list[set[TaskID]]

    def __init__(self, task_groups: Iterable[Iterable[Int]] | None = None):
        if task_groups is None:
            task_groups = []

        self.groups_map = [
            set(TaskID(task_id) for task_id in task_group)
            for task_group in task_groups
        ]

    def add_group(self, task_group: Iterable[Int]) -> None:
        self.groups_map.append(set(TaskID(task_id) for task_id in task_group))

    def remove_group(self, group_id: int) -> None:
        if 0 <= group_id < len(self.groups_map):
            self.groups_map.pop(group_id)

    def initialize(self, state: ScheduleState) -> None:
        if state.debug_mode:
            for tasks in self.groups_map:
                for task in tasks:
                    debug.task_bounds(
                        task,
                        state,
                        "NonOverlapConstraint"
                    )

    def reset(self, state: ScheduleState) -> None:
        self.current_groups = [group.copy() for group in self.groups_map]

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

    def on_pause(self, task_id: TaskID, machine_id: MachineID, state: ScheduleState) -> None:
        for i, group_tasks in enumerate(self.groups_map):
            if task_id not in group_tasks:
                continue

            cur_group_tasks = self.current_groups[i]
            for other_task_id in cur_group_tasks:
                state.reset_bounds(other_task_id)

            self.current_groups[i].add(task_id)

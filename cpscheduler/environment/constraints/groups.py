from typing import Any
from collections.abc import Iterable

from cpscheduler.environment.constants import TaskID, MachineID, Int
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint


class NonOverlapConstraint(Constraint):
    groups_map: list[set[TaskID]]

    current_groups: list[set[TaskID]]

    def __init__(self, task_groups: Iterable[Iterable[Int]]):
        self.groups_map = [
            set(TaskID(task_id) for task_id in task_group)
            for task_group in task_groups
        ]

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.groups_map,),
            (),
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


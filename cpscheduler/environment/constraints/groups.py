from typing import Any
from collections.abc import Iterable

from cpscheduler.utils.list_utils import convert_to_list

from cpscheduler.environment.constants import TASK_ID, Int
from cpscheduler.environment.events import Event
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint

class NonOverlapConstraint(Constraint):
    groups_map: list[set[TASK_ID]]

    current_groups: list[set[TASK_ID]]

    def __init__(self, task_groups: Iterable[Iterable[Int]]):
        self.groups_map = [
            set(convert_to_list(task_group, TASK_ID))
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

    def propagate(self, event: Event, state: ScheduleState) -> None:
        task_id = event.task_id

        if not event.is_assignment():
            return

        for group_tasks in self.groups_map:
            if task_id not in group_tasks:
                continue

            group_tasks.remove(task_id)

            end_time = state.get_end_lb(task_id)

            for other_task_id in group_tasks:
                state.tight_start_lb(other_task_id, end_time)

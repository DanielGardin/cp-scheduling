from collections.abc import Iterable

from cpscheduler.environment.utils import convert_to_list
from cpscheduler.environment.constants import Int, TaskID, MachineID
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import (
    Constraint,
    PassiveConstraint
)

import cpscheduler.environment.debug as debug

class RejectableConstraint(PassiveConstraint):
    """
    A scheduling problem with rejection allows a subset of tasks to be rejected,
    not contributing to the objective function, but usually incurring a penalty
    cost.
    """

    def initialize(self, state: ScheduleState) -> None:
        for task_id in range(state.n_tasks):
            state.instance.set_optionality(task_id)

    def get_entry(self) -> str:
        return "rej"

class AtMostOneConstraint(Constraint):
    """
    Alternative constraint where at most one task have to be processed in each
    one of the groups.
    """

    __slots__ = ("task_groups", "current_tasks")

    task_groups: list[list[TaskID]]

    current_tasks: list[set[TaskID]]

    def __init__(
        self,
        task_groups: Iterable[Iterable[Int]]
    ) -> None:
        self.task_groups = [
            convert_to_list(tasks, TaskID)
            for tasks in task_groups
        ]

        self.current_tasks = []

    def add_group(self, task_group: Iterable[Int]) -> None:
        self.task_groups.append(convert_to_list(task_group, TaskID))

    def remove_group(self, group_id: int) -> None:
        if 0 <= group_id < len(self.task_groups):
            self.task_groups.pop(group_id)

    def initialize(self, state: ScheduleState) -> None:
        for tasks in self.task_groups:
            if not tasks:
                raise ValueError(
                    "Cannot enforce exactly one constraint in a empty set of tasks."
                )

            for task in tasks:
                debug.task_bounds(task, state, type(self).__name__)

    def reset(self, state: ScheduleState) -> None:
        self.current_tasks = [
            set(tasks) for tasks in self.task_groups
        ]

    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            if task_id not in tasks:
                continue

            tasks.remove(task_id)

            for other_task in tasks:
                state.forbid_task(other_task)

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.on_presence(task_id, state)
    
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            tasks.discard(task_id)


class ExactlyOneConstraint(AtMostOneConstraint):
    """
    Alternative constraint where exactly one task have to be processed in each
    one of the groups.
    """

    def reset(self, state: ScheduleState) -> None:
        super().reset(state)

        for tasks in self.current_tasks:
            if len(tasks) == 1:
                remaining_task = next(iter(tasks))
                state.require_task(remaining_task)
                tasks.remove(remaining_task)

    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            if task_id not in tasks:
                continue

            tasks.remove(task_id)

            for other_task in tasks:
                state.forbid_task(other_task)

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.on_presence(task_id, state)

    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        for tasks in self.current_tasks:
            tasks.discard(task_id)

            if len(tasks) == 1:
                remaining_task = next(iter(tasks))
                state.require_task(remaining_task)
                tasks.remove(remaining_task)

            elif len(tasks) == 0:
                state.fail()

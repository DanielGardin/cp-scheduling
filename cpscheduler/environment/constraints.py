from typing import Mapping, Iterable, TypeVar
from copy import deepcopy

from abc import ABC
from textwrap import dedent

from .tasks import Tasks, Status
from .utils import convert_to_list, topological_sort


class Constraint(ABC):
    def set_tasks(self, tasks: Tasks) -> None:
        self.tasks = tasks

    def reset(self) -> None:
        """
        Reset the constraint to its initial state.
        """
        pass

    def propagate(self, time: int) -> None:
        """
        Propagate the constraint at a given time.
        """
        pass

    def export_model(self) -> str:
        return ""

    def export_data(self) -> str:
        return ""


class PrecedenceConstraint(Constraint):
    def __init__(self, precedence: Mapping[int, Iterable[int]]):
        self.precedence = {
            task: convert_to_list(tasks) for task, tasks in precedence.items()
        }

        self.original_precedence = deepcopy(self.precedence)

    def _remove_precedence(self, task: int, child: int) -> None:
        if child in self.precedence[task]:
            self.precedence[task].remove(child)

            if self.topological_order and len(self.precedence[task]) == 0:
                self.topological_order.remove(task)

    def reset(self) -> None:
        self.precedence = deepcopy(self.original_precedence)
        self.topological_order = topological_sort(self.precedence, len(self.tasks))
        self.propagate(0)

    def propagate(self, time: int) -> None:
        ptr = 0

        while ptr < len(self.topological_order):
            task_id = self.topological_order[ptr]
            task = self.tasks[task_id]
            status = task.get_status(time)

            if (
                status == Status.AWAITING or status == Status.PAUSED
            ) and task.get_start_lb() < time:
                task.set_start_lb(time)

            end_time = task.get_end_lb()

            if task_id in self.precedence:
                for child_id in self.precedence[task_id]:
                    child = self.tasks[child_id]

                    if child.is_completed(time):
                        self._remove_precedence(task_id, child_id)
                        continue

                    if child.get_start_lb() < end_time:
                        child.set_start_lb(end_time)

            if ptr + 1 >= len(self.topological_order):
                break

            # Check if the potential removed precedences caused the current task to be removed
            if task_id == self.topological_order[ptr]:
                ptr += 1

    def export_model(self) -> str:
        model = rf"""
            set of int: edges;

            array[edges] of 1..num_tasks: precedence_tasks;
            array[edges] of 1..num_tasks: child_tasks;

            constraint forall(e in edges) (
                end[precedence_tasks[e], num_parts] <= start[child_tasks[e], 1]
            );
        """

        return dedent(model)

    def export_data(self) -> str:
        precedence_tasks: list[str] = []
        child_tasks: list[str] = []

        for task, children in self.original_precedence.items():
            for child in children:
                precedence_tasks.append(str(task + 1))
                child_tasks.append(str(child + 1))

        data = f"""
            edges = 1..{len(precedence_tasks)};
            precedence_tasks = [{', '.join(precedence_tasks)}];
            child_tasks = [{', '.join(child_tasks)}];
        """

        return dedent(data)


_T = TypeVar("_T")


class DisjunctiveConstraint(Constraint):
    def __init__(self, disjunctive_groups: Mapping[_T, Iterable[int]]):
        self.disjunctive_groups = {
            group: convert_to_list(tasks) for group, tasks in disjunctive_groups.items()
        }

        self.original_disjunctive_groups = deepcopy(self.disjunctive_groups)

    def reset(self) -> None:
        self.disjunctive_groups = deepcopy(self.original_disjunctive_groups)

    def propagate(self, time: int) -> None:
        for group, task_ids in self.disjunctive_groups.items():
            minimum_start_time = time

            for task_id in task_ids:
                task = self.tasks[task_id]

                if task.is_fixed():
                    minimum_start_time = max(minimum_start_time, task.get_end_lb())

                if task.is_completed(time):
                    self.disjunctive_groups[group].remove(task_id)

            for task_id in self.disjunctive_groups[group]:
                task = self.tasks[task_id]

                if task.is_fixed():
                    continue

                if task.get_start_lb() < minimum_start_time:
                    task.set_start_lb(minimum_start_time)

    def export_model(self) -> str:
        model = rf"""
            int: num_groups;
            int: num_group_tasks;

            array[1..num_group_tasks] of 1..num_tasks: group_task;
            array[1..num_group_tasks] of 1..num_groups: group;

            constraint forall(g in 1..num_groups) (
                disjunctive([start[group_task[i], 1] | p in 1..num_parts, i in 1..num_group_tasks where group[i] == g],
                            [duration[group_task[i], 1] | p in 1..num_parts, i in 1..num_group_tasks where group[i] == g])
            );
        """

        return dedent(model)

    def export_data(self) -> str:
        group_tasks: list[str] = []
        groups: list[str] = []

        for i, tasks in enumerate(self.original_disjunctive_groups.values(), start=1):
            n_tasks = len(tasks)

            group_tasks.extend([str(task + 1) for task in tasks])
            groups.extend([str(i)] * n_tasks)

        data = f"""\
            num_groups = {len(self.original_disjunctive_groups)};
            num_group_tasks = {len(group_tasks)};
        

            group_task = [{', '.join(group_tasks)}];
            group = [{', '.join(groups)}];
        """

        return dedent(data)


class ReleaseDateConstraint(Constraint):
    def __init__(self, release_dates: Mapping[int, int]):
        self.release_dates = {task: date for task, date in release_dates.items()}

    def reset(self) -> None:
        for task_id, date in self.release_dates.items():
            self.tasks[task_id].set_start_lb(date)


class DeadlineConstraint(Constraint):
    def __init__(self, deadlines: Mapping[int, int]):
        self.deadlines = {task: date for task, date in deadlines.items()}

    def reset(self) -> None:
        for task_id, date in self.deadlines.items():
            self.tasks[task_id].set_end_ub(date)

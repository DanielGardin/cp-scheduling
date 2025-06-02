from typing import Any, Mapping, Iterable, TypeVar
from copy import deepcopy

from abc import ABC
from textwrap import dedent

from .tasks import Tasks, Status
from .utils import convert_to_list, topological_sort, binary_search, is_iterable_type


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

    def get_entry(self) -> str:
        """
        Produce the β entry for the constraint.
        """

        return ""


class PrecedenceConstraint(Constraint):
    def __init__(
        self,
        precedence: Mapping[int, Iterable[int]],
        no_wait: bool = False,
    ):
        self.precedence = {
            task: convert_to_list(tasks) for task, tasks in precedence.items()
        }

        self.no_wait = no_wait

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
        operator = "==" if self.no_wait else "<="

        model = f"""\
            set of int: edges;

            array[edges] of 1..num_tasks: precedence_tasks;
            array[edges] of 1..num_tasks: child_tasks;

            constraint forall(e in edges) (
                end[precedence_tasks[e], num_parts] {operator} start[child_tasks[e], 1]
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

        data = f"""\
            edges = 1..{len(precedence_tasks)};
            precedence_tasks = [{', '.join(precedence_tasks)}];
            child_tasks = [{', '.join(child_tasks)}];
        """

        return dedent(data)

    def get_entry(self) -> str:
        intree = all(len(tasks) <= 1 for tasks in self.precedence.values())

        children = sum([tasks for tasks in self.precedence.values()], [])

        outtree = len(set(children)) == len(children)

        graph = "prec"
        if intree and outtree:
            graph = "chains"

        elif intree:
            graph = "intree"

        elif outtree:
            graph = "outtree"

        if self.no_wait:
            graph += ", nwt"

        return graph


class NoWait(PrecedenceConstraint):
    def __init__(self, precedence: Mapping[int, Iterable[int]]):
        super().__init__(precedence, no_wait=True)


_T = TypeVar("_T")
class DisjunctiveConstraint(Constraint):
    disjunctive_groups: dict[Any, list[int]]

    def __init__(self, disjunctive_groups: Mapping[_T, Iterable[int]] | str):
        if isinstance(disjunctive_groups, str):
            self.disjunctive_groups = {}
            self.tag = disjunctive_groups

        else:
            self.disjunctive_groups = {
                group: convert_to_list(tasks) for group, tasks in disjunctive_groups.items()
            }
            self.tag = ""

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if self.tag:
            for task_id in range(len(tasks)):
                group = tasks.data[self.tag][task_id]

                if group not in self.disjunctive_groups:
                    self.disjunctive_groups[group] = []
                
                self.disjunctive_groups[group].append(task_id)

        self.original_disjunctive_groups = deepcopy(self.disjunctive_groups)

    def reset(self) -> None:
        self.disjunctive_groups = deepcopy(self.original_disjunctive_groups)

    def propagate(self, time: int) -> None:
        for group, task_ids in self.disjunctive_groups.items():
            minimum_start_time = time

            # We go in reverse order to avoid errors when removing tasks
            for i in range(len(task_ids)-1, -1, -1):
                task = self.tasks[task_ids[i]]

                if task.is_fixed():
                    minimum_start_time = max(minimum_start_time, task.get_end_lb())

                if task.is_completed(time):
                    self.disjunctive_groups[group].pop(i)


            for task_id in self.disjunctive_groups[group]:
                task = self.tasks[task_id]

                if task.is_fixed():
                    continue

                if task.get_start_lb() < minimum_start_time:
                    task.set_start_lb(minimum_start_time)

    def export_model(self) -> str:
        model = """\
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
    release_dates: dict[int, int]

    def __init__(self, release_dates: Mapping[int, int] | str):
        if isinstance(release_dates, str):
            self.release_dates = {}
            self.tag = release_dates

        else:
            self.release_dates = {task: date for task, date in release_dates.items()}
            self.tag = ""

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if self.tag:
            date: int
            for task_id in range(len(tasks)):
                date = tasks.data[self.tag][task_id]
                self.release_dates[task_id] = date

        self.original_release_dates = deepcopy(self.release_dates)

    def reset(self) -> None:
        for task_id, date in self.release_dates.items():
            self.tasks[task_id].set_start_lb(date)

    def get_entry(self) -> str:
        return "r_j"


class DeadlineConstraint(Constraint):
    deadlines: dict[int, int]

    def __init__(self, deadlines: Mapping[int, int] | str):
        if isinstance(deadlines, str):
            self.deadlines = {}
            self.tag = deadlines

        else:
            self.deadlines = {task: date for task, date in deadlines.items()}
            self.tag = ""

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if self.tag:
            date: int
            for task_id in range(len(tasks)):
                date = tasks.data[self.tag][task_id]
                self.deadlines[task_id] = date

    def reset(self) -> None:
        for task_id, date in self.deadlines.items():
            self.tasks[task_id].set_end_ub(date)

    def get_entry(self) -> str:
        return "d_j"

class ResourceConstraint(Constraint):
    resources: list[dict[int, float]]

    def __init__(
        self,
        capacities: Iterable[float],
        resource_usage: Iterable[Mapping[int, float]] | Iterable[str],
    ) -> None:
        self.capacities = convert_to_list(capacities)

        if is_iterable_type(resource_usage, str):
            self.resources = [
                {} for _ in range(len(self.capacities))
            ]
            self.tags = resource_usage

        else:
            assert is_iterable_type(resource_usage, dict)

            self.resources = [
                {task: usage for task, usage in resources.items()}
                for resources in resource_usage
            ]
            self.tags = []

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        for i, tag in enumerate(self.tags):
            resource_feature = convert_to_list(tasks.data[tag], float)

            for task_id in range(len(tasks)):
                resource = resource_feature[task_id]

                self.resources[i][task_id] = resource

        self.original_resources = deepcopy(self.resources)

    def reset(self) -> None:
        self.resources = deepcopy(self.original_resources)

    def propagate(self, time: int) -> None:
        for i in range(len(self.resources)):
            task_resources = self.resources[i]

            minimum_end_time: list[int] = []
            resource_taken: list[float] = []
            for task_id in list(task_resources.keys()):
                task = self.tasks[task_id]

                if task.is_executing(time):
                    resource = task_resources[task_id]

                    minimum_end_time.append(task.get_end_lb())
                    resource_taken.append(resource)

                if task.is_completed(time):
                    task_resources.pop(task_id)

            if not resource_taken:
                continue

            argsort = sorted([(end, i) for i, end in enumerate(minimum_end_time)])
            minimum_end_time = [minimum_end_time[i] for _, i in argsort]
            available_resources = resource_taken.copy()

            available_resources[-1] = self.capacities[i] - resource_taken[argsort[-1][1]]

            for i in range(len(minimum_end_time) - 2, -1, -1):
                available_resources[i] = available_resources[i + 1] - resource_taken[argsort[i + 1][1]]

            for task_id in self.resources[i]:
                task     = self.tasks[task_id]
                resource = task_resources[task_id]

                if task.is_fixed():
                    continue

                index = binary_search(available_resources, resource)

                minimum_start_time = minimum_end_time[index - 1] if index > 0 else time

                if task.get_start_lb() < minimum_start_time:
                    task.set_start_lb(minimum_start_time)
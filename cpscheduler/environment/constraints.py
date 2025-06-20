"""
    constraints.py

    This module defines the base class for all constraints in the scheduling environment.
    It provides a common interface for any piece in the scheduling environment that
    interacts with the tasks by limiting when they can be executed, how they are assigned to
    machines, etc.

    You can define your own constraints by subclassing the `Constraint` class and
    implementing the required methods.

"""
from typing import Any, Mapping, Iterable, TypeVar, Optional, Self
from copy import deepcopy

from abc import ABC
import re

from mypy_extensions import mypyc_attr

from .tasks import Tasks, Status
from .utils import convert_to_list, topological_sort, binary_search, is_iterable_type
@mypyc_attr(allow_interpreted_subclasses=True)
class Constraint(ABC):
    """
    Base class for all constraints in the scheduling environment.
    This class provides a common interface for any piece in the scheduling environment that
    interacts with the tasks by limiting when they can be executed, how they are assigned to
    machines, etc.
    """
    tags: dict[str, str]

    tasks: Tasks

    def __init__(self, name: Optional[str] = None) -> None:
        if name is not None and not re.match(r'^[a-zA-Z0-9_]+$', name):
            raise ValueError(
                "Constraint name must be alphanumeric and cannot contain spaces"
                "or special characters."
            )

        self.name = name if name else self.__class__.__name__
        self.loaded = False
        self.tags = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, loaded={self.loaded})"

    def set_tasks(self, tasks: Tasks) -> None:
        "Make the constraint aware of the tasks it is applied to."
        self.tasks = tasks
        self.loaded = True

    def has_tag(self, tag: str) -> bool:
        "Check if the constraint have a tag defined to search on the tasks data."
        has_tag = tag in self.tags

        if has_tag and self.loaded and self.tags[tag] not in self.tasks.data:
            raise ValueError(f"Tag '{tag}' not found in tasks data.")

        return has_tag

    def get_data(self, feature_or_tag: str) -> list[Any]:
        "Get the data for a feature or tag from the tasks data."
        feature = self.tags.get(feature_or_tag, feature_or_tag)

        if feature in self.tasks.data:
            return self.tasks.data[feature]

        if feature in self.tasks.jobs_data:
            data = self.tasks.jobs_data[feature]

            return [data[job] for job in self.tasks.data['job_id']]

        assert False, f"Feature {feature} not found in tasks data"

    def reset(self) -> None:
        "Reset the constraint to its initial state."
        return

    # Some subclasses may not need to implement this method if the constraints
    # are not time-dependent and are ensured in the reset method.
    def propagate(self, time: int) -> None:
        "Propagate the constraint at a given time."
        return

    def get_entry(self) -> str:
        "Produce the β entry for the constraint."
        return ""

class PrecedenceConstraint(Constraint):
    """
    Precedence constraint for the scheduling environment.
    This constraint defines the precedence relationships between tasks, where some tasks
    must be completed before others can start. It can also handle no-wait precedence,
    where tasks must be executed back-to-back without any waiting time in between.

    Arguments:
        precedence: Mapping[int, Iterable[int]]
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 must be completed before task 2 can start,
            the precedence mapping would be {2: [1]}.

        no_wait: bool, default=False
            If True, the precedence constraint will enforce that tasks must be executed
            back-to-back without any waiting time in between. If False, tasks can have a waiting
            time between them.
        
        name: Optional[str] = None
            An optional name for the constraint.
    """
    # original_precedence: dict[int, list[int]]
    precedence: dict[int, list[int]]
    topological_order: list[int]

    def __init__(
        self,
        precedence: Mapping[int, Iterable[int]],
        no_wait: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.original_precedence = {
            task: convert_to_list(tasks) for task, tasks in precedence.items()
        }

        self.no_wait = no_wait

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[int, int]],
        no_wait: bool = False,
        name: Optional[str] = None
    ) -> Self:
        """

        Create a PrecedenceConstraint from a list of edges.

        Arguments:
            edges: Iterable[tuple[int, int]]
                A list of tuples representing the edges of the precedence graph.
                Each tuple (parent, child) indicates that the parent task must be completed
                before the child task can start.

            no_wait: bool, default=False
                If True, the precedence constraint will enforce that tasks must be executed
                back-to-back without any waiting time in between. If False, tasks can have a waiting
                time between them.

            name: Optional[str] = None
                An optional name for the constraint.
        """
        precedence: dict[int, list[int]] = {}

        for parent, child in edges:
            if parent not in precedence:
                precedence[parent] = []

            precedence[parent].append(child)

        return cls(precedence, no_wait, name)

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

            if status in (Status.AWAITING, Status.PAUSED) and task.get_start_lb() < time:
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

    def get_entry(self) -> str:
        n_children = sum(len(tasks) for tasks in self.precedence.values())
        n_unique_children = len(set(
            child for tasks in self.precedence.values() for child in tasks
        ))

        intree  = all(len(tasks) <= 1 for tasks in self.precedence.values())
        outtree = n_children == n_unique_children

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
    """
    
    No-wait precedence constraint for the scheduling environment.
    
    This constraint is a specialized version of the PrecedenceConstraint that enforces
    that tasks must be executed back-to-back without any waiting time in between.
    
    Arguments:
        precedence: Mapping[int, Iterable[int]]
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 must be completed before task 2 can start,
            the precedence mapping would be {2: [1]}.
        
        name: Optional[str] = None
            An optional name for the constraint.
    """

    def __init__(self, precedence: Mapping[int, Iterable[int]], name: Optional[str] = None,):
        super().__init__(precedence, no_wait=True, name=name)


_T = TypeVar("_T")
class DisjunctiveConstraint(Constraint):
    """
    Disjunctive constraint for the scheduling environment.

    This constraint defines disjunctive groups of tasks, where tasks in the same group
    cannot be executed at the same time. The disjunctive groups can be defined as a mapping
    of group names to a list of task IDs, or as a string that refers to a column in the tasks data.

    Arguments:
        disjunctive_groups: Mapping[_T, Iterable[int]] | str
            A mapping of group names to a list of task IDs that belong to the group.
            If a string is provided, it refers to a column in the tasks data that contains
            the group names for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """
    original_disjunctive_groups: dict[Any, list[int]]
    disjunctive_groups: dict[Any, list[int]]

    def __init__(
        self,
        disjunctive_groups: Mapping[_T, Iterable[int]] | str,
        name: Optional[str] = None
    ):
        super().__init__(name)

        if isinstance(disjunctive_groups, str):
            self.tags['disjunctive_groups'] = disjunctive_groups

        else:
            self.original_disjunctive_groups = {
                group: convert_to_list(tasks) for group, tasks in disjunctive_groups.items()
            }

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if self.has_tag('disjunctive_groups'):
            groups = self.get_data("disjunctive_groups")

            self.original_disjunctive_groups = {}
            for task_id in range(len(tasks)):
                group = groups[task_id]

                if group not in self.original_disjunctive_groups:
                    self.original_disjunctive_groups[group] = []

                self.original_disjunctive_groups[group].append(task_id)

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

class ReleaseDateConstraint(Constraint):
    """
    Release date constraint for the scheduling environment.

    This constraint defines the release dates for tasks, which are the earliest times
    that the tasks can start. The release dates can be defined as a mapping of task IDs
    to their respective release dates, or as a string that refers to a column in the tasks data.

    Arguments:
        release_dates: Mapping[int, int] | str
            A mapping of task IDs to their respective release dates. If a string is provided,
            it refers to a column in the tasks data that contains the release dates for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """
    release_dates: dict[int, int]

    def __init__(
            self,
            release_dates: Mapping[int, int] | Iterable[int] | str = 'release_time',
            name: Optional[str] = None
        ):
        super().__init__(name)

        if isinstance(release_dates, str):
            self.tags['release_time'] = release_dates

        elif is_mapping(release_dates, int, int):
            self.release_dates = {task: date for task, date in release_dates.items()}

        else:
            self.release_dates = {task: date for task, date in enumerate(release_dates)}

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if self.has_tag('release_time'):
            release_times = self.get_data("release_time")

            self.release_dates = {
                task_id: release_time for task_id, release_time in enumerate(release_times)
            }

    def reset(self) -> None:
        for task_id, date in self.release_dates.items():
            self.tasks[task_id].set_start_lb(date)

    def get_entry(self) -> str:
        return "r_j"


class DeadlineConstraint(Constraint):
    """
    Deadline constraint for the scheduling environment.

    This constraint defines the deadlines for tasks, which are the latest times
    that the tasks can be completed. The deadlines can be defined as a mapping of task IDs
    to their respective deadlines, or as a string that refers to a column in the tasks data.

    Arguments:
        deadlines: Mapping[int, int] | str
            A mapping of task IDs to their respective deadlines. If a string is provided,
            it refers to a column in the tasks data that contains the deadlines for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """
    deadlines: dict[int, int]

    def __init__(
            self,
            deadlines: Mapping[int, int] | Iterable[int] | str = 'due_dates',
            name: Optional[str] = None
        ):
        super().__init__(name)

        if isinstance(deadlines, str):
            self.tags['due_date'] = deadlines

        elif is_mapping(deadlines, int, int):
            self.deadlines = {task: date for task, date in deadlines.items()}

        else:
            self.deadlines = {task: date for task, date in enumerate(deadlines)}

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if self.has_tag('due_date'):
            due_dates = self.get_data("release_time")

            self.deadlines = {
                task_id: due_date for task_id, due_date in enumerate(due_dates)
            }

    def reset(self) -> None:
        for task_id, date in self.deadlines.items():
            self.tasks[task_id].set_end_ub(date)

    def get_entry(self) -> str:
        return "d_j"


class ResourceConstraint(Constraint):
    """
    Resource constraint for the scheduling environment.

    This constraint defines the resources available for tasks and their usage.
    The resources can be defined as a list of capacities and a list of resource usage for each task.

    Arguments:
        capacities: Iterable[float]
            A list of capacities for each resource. The length of the list should be equal to the
            number of resources.

        resource_usage: Iterable[Mapping[int, float]] | Iterable[str]
            A list of dictionaries or strings that define the resource usage for each task.
            If a string is provided, it refers to a column in the tasks data that contains the
            resource usage for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """
    original_resources: list[dict[int, float]]
    resources: list[dict[int, float]]

    def __init__(
        self,
        capacities: Iterable[float],
        resource_usage: Iterable[Mapping[int, float]] | Iterable[str],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)

        self.capacities = convert_to_list(capacities)

        if is_iterable_type(resource_usage, str):
            for resource_id, resouce_name in enumerate(resource_usage):
                self.tags[f'resource_{resource_id}'] = resouce_name

        else:
            self.original_resources = [
                {task_id: usage for task_id, usage in resources.items()}
                for resources in resource_usage
            ]

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        self.original_resources = [
            {task_id: usage for task_id, usage in enumerate(self.get_data(resource_id))}
            for resource_id in self.tags
        ]

    def reset(self) -> None:
        self.resources = deepcopy(self.original_resources)

    def propagate(self, time: int) -> None:
        for i, task_resources in enumerate(self.resources):
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
                last_task = argsort[i + 1][1]

                available_resources[i] = available_resources[i + 1] - resource_taken[last_task]

            for task_id in self.resources[i]:
                task     = self.tasks[task_id]
                resource = task_resources[task_id]

                if task.is_fixed():
                    continue

                index = binary_search(available_resources, resource)

                minimum_start_time = minimum_end_time[index - 1] if index > 0 else time

                if task.get_start_lb() < minimum_start_time:
                    task.set_start_lb(minimum_start_time)

class MachineConstraint(Constraint):
    """
    General Parallel machine constraint, differs from DisjunctiveConstraint as the disjunctive
    constraint have groups with predefined tasks, while the machine constraint defines its groups
    based on the machine assignment of the tasks.

    Arguments:
        machine_constraint: Iterable[Iterable[int]] | str
            A list of lists of machine ids, each sublist representing the set of
            machines that the corresponding task can be assigned to. The length of the outer list
            should be equal to the number of tasks. Finally, if None is provided, then every task
            can be processed on every machine.
    """
    machine_constraint: list[list[int]]

    def __init__(
        self,
        machine_constraint: Optional[Iterable[Iterable[int]] | str] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)

        self.complete = False
        if isinstance(machine_constraint, str):
            self.tags['machine'] = machine_constraint

        elif machine_constraint is None:
            self.machine_constraint = []
            self.complete = True

        else:
            self.machine_constraint = [
                convert_to_list(tasks) for tasks in machine_constraint
            ]

        # Time when the machine is going to be freed
        self.machine_free: dict[int, int] = {}

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        # str is only implemented for single machine processing constraint
        if self.has_tag('machine'):
            self.machine_constraint = [
                [int(machine)] for machine in self.get_data("machine")
            ]

    def reset(self) -> None:
        self.machine_free.clear()

    def propagate(self, time: int) -> None:
        for task in self.tasks:
            if not task.is_fixed():
                continue

            machine = task.get_assignment()
            end_time = task.get_end()

            if machine not in self.machine_free or end_time > self.machine_free[machine]:
                self.machine_free[machine] = end_time

        for task_id, task in enumerate(self.tasks):
            if task.is_fixed():
                continue

            machines = task.start_bounds if self.complete else self.machine_constraint[task_id]

            for machine in machines:
                if machine not in self.machine_free:
                    self.machine_free[machine] = time

                if task.get_start_lb(machine) < self.machine_free[machine]:
                    task.set_start_lb(self.machine_free[machine], machine)

class SetupConstraint(Constraint):
    """
    Setup constraint for the scheduling environment.

    This constraint is used to define the setup time between tasks.
    The setup times can be defined as a mapping of task IDs to a mapping of child task IDs
    and their respective setup times, or as a string that refers to a column in the tasks data.

    Arguments:
        setup_times: Mapping[int, Mapping[int, int]] | str
            A mapping of task IDs to a mapping of child task IDs and their respective setup times.
            If a string is provided, it refers to a column in the tasks data that contains the setup
            times.

        name: Optional[str] = None
            An optional name for the constraint.
    """
    setup_times: dict[int, dict[int, int]]

    def __init__(
            self,
            setup_times: Mapping[int, Mapping[int, int]],
            name: Optional[str] = None
        ) -> None:
        super().__init__(name)

        self.original_setup_times = {
            task: {child: time for child, time in children.items()}
            for task, children in setup_times.items()
        }

    def reset(self) -> None:
        self.setup_times = deepcopy(self.original_setup_times)

    def propagate(self, time: int) -> None:
        for task_id in list(self.setup_times.keys()):
            task = self.tasks[task_id]

            if task.is_completed(time):
                self.setup_times.pop(task_id)
                continue

            if not task.is_fixed():
                continue

            children = self.setup_times[task_id]

            for child_id, setup_time in children.items():
                child = self.tasks[child_id]

                if child.is_fixed():
                    continue

                if task.get_end_lb() + setup_time > child.get_start_lb():
                    child.set_start_lb(task.get_end_lb() + setup_time)

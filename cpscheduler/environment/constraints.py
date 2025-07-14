"""
constraints.py

This module defines the base class for all constraints in the scheduling environment.
It provides a common interface for any piece in the scheduling environment that
interacts with the tasks by limiting when they can be executed, how they are assigned to
machines, etc.

You can define your own constraints by subclassing the `Constraint` class and
implementing the required methods.

"""

from typing import Any
from collections.abc import Iterable, Mapping, Sequence
from typing_extensions import Self

import re

from mypy_extensions import mypyc_attr

from ._common import TASK_ID, TIME, Int, Float
from .data import SchedulingData
from .tasks import Tasks
from .utils import convert_to_list, topological_sort, binary_search, is_iterable_type

constraints: dict[str, type["Constraint"]] = {}


@mypyc_attr(allow_interpreted_subclasses=True)
class Constraint:
    """
    Base class for all constraints in the scheduling environment.
    This class provides a common interface for any piece in the scheduling environment that
    interacts with the tasks by limiting when they can be executed, how they are assigned to
    machines, etc.
    """

    tags: dict[str, str]
    setup_constraint: bool

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        constraints[cls.__name__] = cls

    def __init__(self, name: str | None = None) -> None:
        if name is not None and not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError(
                "Constraint name must be alphanumeric and cannot contain spaces"
                "or special characters."
            )

        self.name = name if name else self.__class__.__name__
        self.tags = {}
        self.setup_constraint = False

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}()"

    def import_data(self, data: SchedulingData) -> None:
        "Import data from the instance when necessary."

    def export_data(self, data: SchedulingData) -> None:
        "Export data to the instance when necessary."

    def reset(self, tasks: Tasks) -> None:
        "Reset the constraint to its initial state."

    def propagate(self, time: TIME, tasks: Tasks) -> None:
        "Ensure the constraint is satisfied."

    def get_entry(self) -> str:
        "Produce the β entry for the constraint."
        return ""

    def to_dict(self) -> dict[str, Any]:
        "Serialize the objective to a dictionary."
        raise NotImplementedError(
            f"{self.__class__.__name__} serialization is not implemented."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        "Deserialize the objective from a dictionary."
        return cls(**data)


class MachineConstraint(Constraint):
    """
    General Parallel machine constraint, differs from DisjunctiveConstraint as the disjunctive
    constraint have groups with predefined tasks, while the machine constraint defines its groups
    based on the machine assignment of the tasks.
    """

    machine_free: list[TIME]

    def import_data(self, data: SchedulingData) -> None:
        self.machine_free = [0 for _ in range(data.n_machines)]

    def reset(self, tasks: Tasks) -> None:
        for machine, _ in enumerate(self.machine_free):
            self.machine_free[machine] = 0

    def propagate(self, time: TIME, tasks: Tasks) -> None:
        for task_id in tasks.transition_tasks:
            task = tasks[task_id]

            machine = task.get_assignment()
            end_time = task.get_end()

            self.machine_free[machine] = end_time

        for task_id in tasks.awaiting_tasks:
            task = tasks[task_id]

            for machine in task.machines:
                if task.get_start_lb(machine) < self.machine_free[machine]:
                    task.set_start_lb(self.machine_free[machine], machine)

    def is_complete(self, tasks: Tasks) -> bool:
        "Check if the machine constraint is complete."
        n_machines = len(self.machine_free)
        return all(len(task.machines) == n_machines for task in tasks)


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
    precedence: dict[TASK_ID, set[TASK_ID]]
    original_order: list[TASK_ID]

    def __init__(
        self,
        precedence: Mapping[Int, Sequence[Int]],
        no_wait: bool = False,
        name: str | None = None,
    ):
        super().__init__(name)

        self.precedence = {
            TASK_ID(task): {TASK_ID(child) for child in children}
            for task, children in precedence.items()
        }

        self.no_wait = no_wait

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[Int, Int]],
        no_wait: bool = False,
        name: str | None = None,
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
        precedence: dict[Int, list[Int]] = {}

        for parent, child in edges:
            if parent not in precedence:
                precedence[parent] = []

            precedence[parent].append(child)

        return cls(precedence, no_wait, name)

    def is_intree(self) -> bool:
        "Check if the precedence graph is an in-tree."
        for tasks in self.precedence.values():
            if len(tasks) > 1:
                return False

        return True

    def is_outtree(self) -> bool:
        "Check if the precedence graph is an out-tree."
        n_children = 0
        unique_children: set[TASK_ID] = set()

        for tasks in self.precedence.values():
            n_children += len(tasks)
            unique_children.update(tasks)

        return n_children == len(unique_children)

    def import_data(self, data: SchedulingData) -> None:
        self.original_order = topological_sort(self.precedence, data.n_tasks)

    def export_data(self, data: SchedulingData) -> None:
        if self.is_intree():
            successors = [
                (
                    next(iter(self.precedence[task_id]))
                    if task_id in self.precedence
                    else -1
                )
                for task_id in range(data.n_tasks)
            ]

            data.add_data("successor", successors)

        if self.is_outtree():
            predecessors = [-1 for _ in range(data.n_tasks)]
            for task_id, children in self.precedence.items():
                for child_id in children:
                    predecessors[child_id] = task_id

            data.add_data("predecessor", predecessors)

    def reset(self, tasks: Tasks) -> None:
        self.topological_order = self.original_order.copy()

    def propagate(self, time: TIME, tasks: Tasks) -> None:
        for task_id in list(self.topological_order):
            task = tasks[task_id]

            end_time = task.get_end_lb()

            for child_id in self.precedence[task_id]:
                child = tasks[child_id]

                if child.get_start_lb() < end_time:
                    child.set_start_lb(end_time)

            if task.is_completed(time):
                self.topological_order.remove(task_id)

    def get_entry(self) -> str:
        intree = self.is_intree()
        outtree = self.is_outtree()

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

    def __init__(
        self,
        precedence: Mapping[Int, Sequence[Int]],
        name: str | None = None,
    ):
        super().__init__(precedence, no_wait=True, name=name)


class ConstantProcessingTime(Constraint):
    """
    Constant processing time constraint for the scheduling environment.

    This constraint enforces that all tasks have the same processing time, which is defined
    as a constant value.

    Arguments:
        processing_time: int
            The constant processing time for all tasks.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    def __init__(self, processing_time: Int, name: str | None = None):
        super().__init__(name)
        self.processing_time = TIME(processing_time)

    def export_data(self, data: SchedulingData) -> None:
        for task_id in range(data.n_tasks):
            for machine in data.processing_times[task_id]:
                data.processing_times[task_id][machine] = self.processing_time


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

    original_disjunctive_groups: dict[Any, list[TASK_ID]]

    def __init__(
        self,
        disjunctive_groups: Mapping[Any, Iterable[Int]] | str,
        name: str | None = None,
    ):
        super().__init__(name)

        if isinstance(disjunctive_groups, str):
            self.tags["disjunctive_groups"] = disjunctive_groups

        else:
            self.original_disjunctive_groups = {
                group: convert_to_list(tasks, TASK_ID)
                for group, tasks in disjunctive_groups.items()
            }

    def import_data(self, data: SchedulingData) -> None:
        if "disjunctive_groups" in self.tags:
            groups = data.get_task_level_data(self.tags["disjunctive_groups"])

            self.original_disjunctive_groups = {}
            for task_id in range(data.n_tasks):
                group = groups[task_id]

                if group not in self.original_disjunctive_groups:
                    self.original_disjunctive_groups[group] = []

                self.original_disjunctive_groups[group].append(task_id)

    def reset(self, tasks: Tasks) -> None:
        self.disjunctive_groups = {
            group: group_tasks.copy()
            for group, group_tasks in self.original_disjunctive_groups.items()
        }

    def propagate(self, time: TIME, tasks: Tasks) -> None:
        for group, task_ids in self.disjunctive_groups.items():
            minimum_start_time = time

            # We go in reverse order to avoid errors when removing tasks
            for i in range(len(task_ids) - 1, -1, -1):
                task = tasks[task_ids[i]]

                if task.is_fixed():
                    end_lb = task.get_end_lb()
                    if end_lb > minimum_start_time:
                        minimum_start_time = end_lb

                if task.is_completed(time):
                    self.disjunctive_groups[group].pop(i)

            for task_id in self.disjunctive_groups[group]:
                task = tasks[task_id]

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

    release_dates: dict[TASK_ID, TIME]

    def __init__(
        self,
        release_dates: Mapping[Int, Int] | Iterable[Int] | str = "release_time",
        name: str | None = None,
    ):
        super().__init__(name)

        if isinstance(release_dates, str):
            self.tags["release_time"] = release_dates

        elif isinstance(release_dates, Mapping):
            self.release_dates = {
                TASK_ID(task): TIME(date) for task, date in release_dates.items()
            }

        else:
            self.release_dates = {
                TASK_ID(task): TIME(date) for task, date in enumerate(release_dates)
            }

    def import_data(self, data: SchedulingData) -> None:
        if "release_time" in self.tags:
            release_times = data.get_task_level_data(self.tags["release_time"])

            self.release_dates = {
                TASK_ID(task_id): TIME(release_time)
                for task_id, release_time in enumerate(release_times)
            }

    def export_data(self, data: SchedulingData) -> None:
        if not self.release_dates:
            data.add_data(
                "release_time",
                [
                    self.release_dates.get(TASK_ID(task), TIME(0))
                    for task in range(data.n_tasks)
                ],
            )

        else:
            data.add_alias("release_time", self.tags["release_time"])

    def reset(self, tasks: Tasks) -> None:
        for task_id, date in self.release_dates.items():
            tasks[task_id].set_start_lb(date)

    def get_entry(self) -> str:
        return "r_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_dates": self.tags.get("release_time", "release_time"),
            "name": self.name,
        }


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

    deadlines: dict[TASK_ID, TIME]

    def __init__(
        self,
        deadlines: Mapping[Int, Int] | Iterable[Int] | str = "due_dates",
        name: str | None = None,
    ):
        super().__init__(name)

        if isinstance(deadlines, str):
            self.tags["due_date"] = deadlines

        elif isinstance(deadlines, Mapping):
            self.deadlines = {
                TASK_ID(task): TIME(date) for task, date in deadlines.items()
            }

        else:
            self.deadlines = {
                TASK_ID(task): TIME(date) for task, date in enumerate(deadlines)
            }

    def import_data(self, data: SchedulingData) -> None:
        if "due_date" in self.tags:
            due_dates = data.get_task_level_data(self.tags["due_date"])

            self.deadlines = {
                TASK_ID(task_id): TIME(due_date)
                for task_id, due_date in enumerate(due_dates)
            }

    def export_data(self, data: SchedulingData) -> None:
        if not self.deadlines:
            data.add_data(
                "due_date",
                [
                    self.deadlines.get(TASK_ID(task), TIME(0))
                    for task in range(data.n_tasks)
                ],
            )

        else:
            data.add_alias("due_date", self.tags["due_date"])

    def reset(self, tasks: Tasks) -> None:
        for task_id, date in self.deadlines.items():
            tasks[task_id].set_end_ub(date)

    def get_entry(self) -> str:
        return "d_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "deadlines": self.tags.get("due_date", "due_date"),
            "name": self.name,
        }


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

    capacities: list[float]
    original_resources: list[dict[TASK_ID, float]]

    def __init__(
        self,
        capacities: Iterable[Float],
        resource_usage: Iterable[Mapping[Int, Float]] | Iterable[str],
        name: str | None = None,
    ) -> None:
        super().__init__(name)

        self.capacities = convert_to_list(capacities, float)

        if is_iterable_type(resource_usage, str):
            for resource_id, resouce_name in enumerate(resource_usage):
                self.tags[f"resource_{resource_id}"] = resouce_name

        else:
            self.original_resources = [
                {TASK_ID(task_id): float(usage) for task_id, usage in resources.items()}
                for resources in resource_usage
            ]

    def import_data(self, data: SchedulingData) -> None:
        self.original_resources = [
            {
                TASK_ID(task_id): float(usage)
                for task_id, usage in enumerate(
                    data.get_task_level_data(self.tags[f"resource_{resource_id}"])
                )
            }
            for resource_id in self.tags
        ]

    def reset(self, tasks: Tasks) -> None:
        self.resources = [resources.copy() for resources in self.original_resources]

    def propagate(self, time: TIME, tasks: Tasks) -> None:
        for i, task_resources in enumerate(self.resources):
            minimum_end_time: list[TIME] = []
            resource_taken: list[float] = []
            for task_id in list(task_resources.keys()):
                task = tasks[task_id]

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

            available_resources[-1] = (
                self.capacities[i] - resource_taken[argsort[-1][1]]
            )

            for i in range(len(minimum_end_time) - 2, -1, -1):
                last_task = argsort[i + 1][1]

                available_resources[i] = (
                    available_resources[i + 1] - resource_taken[last_task]
                )

            for task_id in self.resources[i]:
                task = tasks[task_id]
                resource = task_resources[task_id]

                if task.is_fixed():
                    continue

                index = binary_search(available_resources, resource)

                minimum_start_time = minimum_end_time[index - 1] if index > 0 else time

                if task.get_start_lb() < minimum_start_time:
                    task.set_start_lb(minimum_start_time)


# TODO: Check literature if the setup time only happens when in the same machine
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

    original_setup_times: dict[TASK_ID, dict[TASK_ID, TIME]]

    def __init__(
        self,
        setup_times: Mapping[Int, Mapping[Int, Int]],
        name: str | None = None,
    ) -> None:
        super().__init__(name)

        self.original_setup_times = {
            TASK_ID(task): {
                TASK_ID(child): TIME(time) for child, time in children.items()
            }
            for task, children in setup_times.items()
        }

    def reset(self, tasks: Tasks) -> None:
        self.setup_times = {
            task_id: children.copy()
            for task_id, children in self.original_setup_times.items()
        }

    def propagate(self, time: TIME, tasks: Tasks) -> None:
        for task_id in list(self.setup_times.keys()):
            task = tasks[task_id]

            if task.is_completed(time):
                self.setup_times.pop(task_id)
                continue

            if not task.is_fixed():
                continue

            children = self.setup_times[task_id]

            for child_id, setup_time in children.items():
                child = tasks[child_id]

                if child.is_fixed():
                    continue

                if task.get_end_lb() + setup_time > child.get_start_lb():
                    child.set_start_lb(task.get_end_lb() + setup_time)

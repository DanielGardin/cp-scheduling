"""
constraints.py

This module defines the base class for all constraints in the scheduling environment.
It provides a common interface for any piece in the scheduling environment that
interacts with the tasks by limiting when they can be executed, how they are assigned to
machines, etc.

You can define your own constraints by subclassing the `Constraint` class and
implementing the required methods.

"""

from typing import Any, TypeAlias
from collections.abc import Iterable, Mapping, Sequence, Callable
from typing_extensions import Self

import re

from mypy_extensions import mypyc_attr

from cpscheduler.utils.list_utils import convert_to_list
from cpscheduler.utils.general_algo import topological_sort, binary_search

from cpscheduler.environment._common import TASK_ID, TIME, MACHINE_ID, Int, Float
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.tasks import Task


constraints: dict[str, type["Constraint"]] = {}


@mypyc_attr(allow_interpreted_subclasses=True)
class Constraint:
    """
    Base class for all constraints in the scheduling environment.
    This class provides a common interface for any piece in the scheduling environment that
    interacts with the tasks by limiting when they can be executed, how they are assigned to
    machines, etc.
    """

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

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}()"

    def initialize(self, state: ScheduleState) -> None:
        "Initialize the constraint with the scheduling state."

    def reset(self, state: ScheduleState) -> None:
        "Reset the constraint to its initial state."

    def propagate(self, time: TIME, state: ScheduleState) -> None:
        "Given a bound change, propagate the constraint to other tasks."

    def refresh(self, time: TIME, state: ScheduleState) -> None:
        "Updates constraint internal state when propagate cannot handle."

    def get_entry(self) -> str:
        "Produce the β entry for the constraint."
        return ""


class PreemptionConstraint(Constraint):
    """
    Preemption constraint for the scheduling environment.
    This constraint allows tasks to be preempted, meaning they can be interrupted
    and resumed later.

    Arguments:
        name: Optional[str] = None
            An optional name for the constraint.

    Note:
        This constraint is a placeholder and does not implement any specific logic
        for preemption. It serves as a marker to indicate that preemption is allowed
        in the scheduling environment, following the convention used in scheduling literature.

        Another way to provide to the environment that preemption is allowed is to set
        the `allow_preemption` flag in the `SchedulingEnv` initialization.
    """

    task_ids: list[TASK_ID]
    all_tasks: bool

    def __init__(
        self,
        task_ids: Iterable[Int] | None = None,
        name: str | None = None
    ) -> None:
        super().__init__(name)

        if task_ids is None:
            self.all_tasks = True
            self.task_ids = []

        else:
            self.all_tasks = False
            self.task_ids = convert_to_list(task_ids, TASK_ID)

    def initialize(self, state: ScheduleState) -> None:
        if self.all_tasks:
            for task in state.tasks:
                task.set_preemption(True)

        else:
            for task_id in self.task_ids:
                state.tasks[task_id].set_preemption(True)

    def get_entry(self) -> str:
        return "prmp"


class MachineConstraint(Constraint):
    """
    General Parallel machine constraint, differs from DisjunctiveConstraint as the disjunctive
    constraint have groups with predefined tasks, while the machine constraint defines its groups
    based on the machine assignment of the tasks.
    """

    machine_free: list[TIME]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.machine_free = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.name,),
            (self.machine_free,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.machine_free,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.machine_free = [0 for _ in range(state.n_machines)]

    def reset(self, state: ScheduleState) -> None:
        for machine, _ in enumerate(self.machine_free):
            self.machine_free[machine] = 0

    def propagate(self, time: TIME, state: ScheduleState) -> None:
        changed_machines: set[MACHINE_ID] = set()
        for task in state.transition_tasks:
            machine = task.get_assignment()

            self.machine_free[machine] = task.get_end()
            changed_machines.add(machine)

        for task in state.awaiting_tasks:
            for machine in changed_machines:
                if machine not in task.machines:
                    continue

                if task.get_start_lb(machine) < self.machine_free[machine]:
                    task.set_start_lb(self.machine_free[machine], machine)

    def refresh(self, time: TIME, state: ScheduleState) -> None:
        for machine in range(len(self.machine_free)):
            self.machine_free[machine] = time

        for task in state.fixed_tasks:
            for part in range(task.n_parts):
                machine = task.get_assignment(part)

                if task.get_end(part) > self.machine_free[machine]:
                    self.machine_free[machine] = task.get_end(part)

    def is_complete(self, state: ScheduleState) -> bool:
        "Check if the machine constraint is complete."
        n_machines = len(self.machine_free)
        return all(len(task.machines) == n_machines for task in state.tasks)

    def get_entry(self) -> str:
        return "M_j"


class PrecedenceConstraint(Constraint):
    """
    Precedence constraint for the scheduling environment.
    This constraint defines the precedence relationships between tasks, where some tasks
    must be completed before others can start.

    Arguments:
        precedence: Mapping[int, Iterable[int]]
            A mapping of task IDs to a list of task IDs that must be completed before
            the task can start. For example, if task 1 must be completed before task 2 can start,
            the precedence mapping would be {2: [1]}.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    precedence: dict[TASK_ID, list[TASK_ID]]

    original_order: list[TASK_ID]
    tasks_order: list[Task]

    def __init__(
        self,
        precedence: Mapping[Int, Sequence[Int]],
        name: str | None = None,
    ):
        super().__init__(name)

        self.precedence = {
            TASK_ID(task): [TASK_ID(child) for child in children]
            for task, children in precedence.items()
        }

        self.original_order = []
        self.tasks_order = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.precedence, self.name),
            (self.original_order, self.tasks_order),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.original_order, self.tasks_order = state

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[Int, Int]],
        name: str | None = None,
    ) -> Self:
        """
        Create a PrecedenceConstraint from a list of edges.

        Arguments:
            edges: Iterable[tuple[int, int]]
                A list of tuples representing the edges of the precedence graph.
                Each tuple (parent, child) indicates that the parent task must be completed
                before the child task can start.

            name: Optional[str] = None
                An optional name for the constraint.
        """
        precedence: dict[Int, list[Int]] = {}

        for parent, child in edges:
            if parent not in precedence:
                precedence[parent] = []

            precedence[parent].append(child)

        return cls(precedence, name)

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

    def initialize(self, state: ScheduleState) -> None:
        self.original_order = topological_sort(self.precedence, state.n_tasks)

    def reset(self, state: ScheduleState) -> None:
        self.tasks_order = [state.tasks[task_id] for task_id in self.original_order]

    def propagate(self, time: TIME, state: ScheduleState) -> None:
        completion_mask = [task.is_completed(time) for task in self.tasks_order]

        if any(completion_mask):
            self.tasks_order = [
                task
                for i, task in enumerate(self.tasks_order)
                if not completion_mask[i]
            ]

        for task in self.tasks_order:
            end_time = task.get_end_lb()

            for child_id in self.precedence.get(task.task_id, []):
                child = state.tasks[child_id]

                if child.get_start_lb() < end_time:
                    child.set_start_lb(end_time)

    def refresh(self, time: TIME, state: ScheduleState) -> None:
        self.reset(state)

        for task in list(self.tasks_order):
            if task.is_completed(time):
                self.tasks_order.remove(task)

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

        return graph


class NoWaitConstraint(PrecedenceConstraint):
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
        super().__init__(precedence, name=name)

        if not self.is_intree():
            raise ValueError("No-wait constraint must be an in-tree.")

    def propagate(self, time: TIME, state: ScheduleState) -> None:
        super().propagate(time, state)

        for task in reversed(self.tasks_order):
            if task.is_fixed():
                end_time = task.get_end_lb()

                for child_id in self.precedence.get(task.task_id, []):
                    state.tasks[child_id].set_start_ub(end_time)

            else:
                max_children_start = task.get_end_lb()
                for child_id in self.precedence.get(task.task_id, []):
                    child = state.tasks[child_id]

                    child_lb = child.get_start_lb()

                    if max_children_start < child_lb:
                        max_children_start = child_lb

                task.set_end_lb(max_children_start)

    def get_entry(self) -> str:
        return "nwt"


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

    def __init__(self, processing_time: Int = 1, name: str | None = None):
        super().__init__(name)
        self.processing_time = TIME(processing_time)

    def initialize(self, state: ScheduleState) -> None:
        for task in state.tasks:
            for machine in task.machines:
                task.set_processing_time(machine, self.processing_time)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.processing_time, self.name),
            (),
        )

    def get_entry(self) -> str:
        return f"p_j={self.processing_time}"


class DisjunctiveConstraint(Constraint):
    groups_map: dict[TASK_ID, list[int]]

    group_free: list[TIME]

    def __init__(
        self,
        task_groups: Iterable[Iterable[Int]],
        name: str | None = None,
    ):
        super().__init__(name)

        self.group_free = [0 for _ in task_groups]
        self.groups_map = {}

        for group_id, group in enumerate(task_groups):
            for task in group:
                task_id = TASK_ID(task)

                self.groups_map.setdefault(task_id, [])
                self.groups_map[task_id].append(group_id)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.groups_map, self.name),
            (self.tags, self.group_free),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.tags, self.group_free = state

    def reset(self, state: ScheduleState) -> None:
        for i in range(len(self.group_free)):
            self.group_free[i] = 0

    def propagate(self, time: TIME, state: ScheduleState) -> None:
        for task in state.transition_tasks:
            for group in self.groups_map[task.task_id]:
                self.group_free[group] = task.get_end()

        for task in state.awaiting_tasks:
            for group in self.groups_map[task.task_id]:
                if task.get_start_lb(group) < self.group_free[group]:
                    task.set_start_lb(self.group_free[group], group)

    def refresh(self, time: TIME, state: ScheduleState) -> None:
        for group in self.group_free:
            self.group_free[group] = time

        for task in state.fixed_tasks:
            for part in range(task.n_parts):
                for group in self.groups_map[task.task_id]:
                    if task.get_end(part) > self.group_free[group]:
                        self.group_free[group] = task.get_end(part)


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

    release_tag: str
    release_dates: dict[TASK_ID, TIME]

    def __init__(
        self,
        release_dates: str = "release_time",
        name: str | None = None,
    ):
        super().__init__(name)
        self.release_tag = release_dates

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.release_dates, self.name),
            (self.tags,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.tags,) = state

    def initialize(self, state: ScheduleState) -> None:
        release_times = state.instance[self.release_tag]

        self.release_dates = {
            TASK_ID(task_id): TIME(release_time)
            for task_id, release_time in enumerate(release_times)
        }

    def reset(self, state: ScheduleState) -> None:
        for task_id, release_time in self.release_dates.items():
            state.tasks[task_id].set_start_lb(release_time)

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

    due_tag: str
    due_dates: dict[TASK_ID, TIME]

    def __init__(
        self,
        due_dates: str = "due_dates",
        name: str | None = None,
    ):
        super().__init__(name)
        self.due_tag = due_dates

    def initialize(self, state: ScheduleState) -> None:
        due_times = state.instance[self.due_tag]

        self.due_dates = {
            TASK_ID(task_id): TIME(due_time)
            for task_id, due_time in enumerate(due_times)
        }

    def reset(self, state: ScheduleState) -> None:
        for task_id, due_time in self.due_dates.items():
            state.tasks[task_id].set_end_ub(due_time)

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

    resource_tags: list[str]
    capacities: list[float]
    original_resources: list[dict[TASK_ID, float]]

    resources: list[dict[TASK_ID, float]]

    def __init__(
        self,
        capacities: Iterable[Float],
        resource_usage: Iterable[str],
        name: str | None = None,
    ) -> None:
        super().__init__(name)

        self.capacities = convert_to_list(capacities, float)
        self.resource_tags = list(resource_usage)

    def initialize(self, state: ScheduleState) -> None:
        self.original_resources = [
            {
                TASK_ID(task_id): float(usage)
                for task_id, usage in enumerate(state.instance[resource_tag])
            }
            for resource_tag in self.resource_tags
        ]

    def reset(self, state: ScheduleState) -> None:
        self.resources = [resources.copy() for resources in self.original_resources]

    def propagate(self, time: TIME, state: ScheduleState) -> None:
        for i, task_resources in enumerate(self.resources):
            minimum_end_time: list[TIME] = []
            resource_taken: list[float] = []
            for task_id in list(task_resources.keys()):
                task = state.tasks[task_id]

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
                task = state.tasks[task_id]
                resource = task_resources[task_id]

                if task.is_fixed():
                    continue

                index = binary_search(available_resources, resource)

                minimum_start_time = minimum_end_time[index - 1] if index > 0 else time

                if task.get_start_lb() < minimum_start_time:
                    task.set_start_lb(minimum_start_time)

    def refresh(self, time: TIME, state: ScheduleState) -> None:
        raise NotImplementedError(
            "Refresh method is not implemented for ResourceConstraint yet."
        )


SetupTimes: TypeAlias = Mapping[Int, Mapping[Int, Int]] | Callable[[int, int, Any], Int]


# TODO: Check literature if the setup time only happens when in the same machine
class SetupConstraint(Constraint):
    """
    Setup constraint for the scheduling environment.

    This constraint is used to define the setup time between tasks.
    The setup times can be defined as a mapping of task IDs to a mapping of child task IDs
    and their respective setup times, or as a string that refers to a column in the tasks data.

    Arguments:
        setup_times: Mapping[int, Mapping[int, int]] | Callable[[int, int, ScheduleState], int]
            A mapping of task IDs to a mapping of child task IDs and their respective setup times.
            Alternatively, a callable function that takes in two task IDs and the scheduling data,
            and returns the setup time between the two tasks.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    original_setup_times: dict[TASK_ID, dict[TASK_ID, TIME]]
    setup_fn: Callable[[int, int, ScheduleState], Int] | None = None

    def __init__(
        self,
        setup_times: SetupTimes,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        if callable(setup_times):
            self.setup_fn = setup_times
            self.original_setup_times = {}

        else:
            self.original_setup_times = {
                TASK_ID(task): {
                    TASK_ID(child): TIME(time) for child, time in children.items()
                }
                for task, children in setup_times.items()
            }

    def initialize(self, state: ScheduleState) -> None:
        if self.setup_fn is not None:
            self.original_setup_times = {}

            for task_id in range(state.n_tasks):
                self.original_setup_times[TASK_ID(task_id)] = {}

                for child_id in range(state.n_tasks):
                    if task_id == child_id:
                        continue

                    setup_time = TIME(self.setup_fn(task_id, child_id, state))

                    if setup_time > 0:
                        self.original_setup_times[TASK_ID(task_id)][
                            TASK_ID(child_id)
                        ] = setup_time

    def reset(self, state: ScheduleState) -> None:
        self.setup_times = {
            task_id: children.copy()
            for task_id, children in self.original_setup_times.items()
        }

    def propagate(self, time: TIME, state: ScheduleState) -> None:
        for task_id in list(self.setup_times.keys()):
            task = state.tasks[task_id]

            if task.is_completed(time):
                self.setup_times.pop(task_id)
                continue

            if not task.is_fixed():
                continue

            children = self.setup_times[task_id]

            for child_id, setup_time in children.items():
                child = state.tasks[child_id]

                if child.is_fixed():
                    continue

                if task.get_end_lb() + setup_time > child.get_start_lb():
                    child.set_start_lb(task.get_end_lb() + setup_time)

    def refresh(self, time: TIME, state: ScheduleState) -> None:
        raise NotImplementedError(
            "Refresh method is not implemented for SetupConstraint yet."
        )

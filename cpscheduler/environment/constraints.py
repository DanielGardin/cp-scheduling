"""
constraints.py

This module defines the base class for all constraints in the scheduling environment.
It provides a common interface for any piece in the scheduling environment that
interacts with the tasks by limiting when they can be executed, how they are assigned to
machines, etc.

You can define your own constraints by subclassing the `Constraint` class and
implementing the required methods.

"""

from typing import Any, TypeAlias, NoReturn
from collections.abc import Iterable, Mapping, Sequence, Callable
from typing_extensions import Self

from mypy_extensions import mypyc_attr

from cpscheduler.utils.list_utils import convert_to_list
from cpscheduler.utils.general_algo import binary_search, topological_sort

from cpscheduler.environment._common import (
    TASK_ID,
    TIME,
    MACHINE_ID,
    Int,
    Float,
    MAX_TIME,
    GLOBAL_MACHINE_ID,
)
from cpscheduler.environment.events import Event
from cpscheduler.environment.state import ScheduleState

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

    def initialize(self, state: ScheduleState) -> None:
        "Initialize the constraint with the scheduling state."

    def reset(self, state: ScheduleState) -> None:
        "Reset the constraint to its initial state."

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        "Given a bound change, propagate the constraint to other tasks."

    def get_entry(self) -> str:
        "Produce the β entry for the constraint."
        return ""


class PassiveConstraint(Constraint):
    """
    Passive constraints do not actively propagate changes in the scheduling environment, not being
    directly translatable into mathematical programming constraints, but rather serve as markers or
    flags to indicate certain properties of tasks.

    Examples of passive constraints include preemption and optionality constraints.
    """

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> NoReturn:
        "Passive constraint does not propagate any changes."
        raise NotImplementedError("Passive constraint does not propagate any changes.")

    def reset(self, state: ScheduleState) -> NoReturn:
        "Passive constraint does not reset any state."
        raise NotImplementedError("Passive constraint does not reset any state.")


class PreemptionConstraint(PassiveConstraint):
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

    def __init__(self, task_ids: Iterable[Int] | None = None) -> None:
        if task_ids is None:
            self.all_tasks = True
            self.task_ids = []

        else:
            self.all_tasks = False
            self.task_ids = convert_to_list(task_ids, TASK_ID)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.task_ids,),
            (),
        )

    def initialize(self, state: ScheduleState) -> None:
        if self.all_tasks:
            for task in state.tasks:
                task.set_preemption(True)

        else:
            for task_id in self.task_ids:
                state.tasks[task_id].set_preemption(True)

    def get_entry(self) -> str:
        return "prmp"


class OptionalityConstraint(PassiveConstraint):
    """
    Makes tasks optional in the scheduling environment.
    Tasks marked as optional are treated equally to regular tasks, but they can be
    left unscheduled without affecting the feasibility of the overall schedule.

    Arguments:
        task_ids: Iterable[int] | None
            A list of task IDs to be marked as optional. If None, all tasks are marked as optional.
    """

    task_ids: list[TASK_ID]
    all_tasks: bool

    def __init__(self, task_ids: Iterable[Int] | None = None) -> None:
        if task_ids is None:
            self.all_tasks = True
            self.task_ids = []

        else:
            self.all_tasks = False
            self.task_ids = convert_to_list(task_ids, TASK_ID)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.task_ids,),
            (),
        )

    def initialize(self, state: ScheduleState) -> None:
        if self.all_tasks:
            for task in state.tasks:
                task.set_optionality(True)

        else:
            for task_id in self.task_ids:
                state.tasks[task_id].set_optionality(True)

    def get_entry(self) -> str:
        return "opt"


class MachineEligibilityConstraint(PassiveConstraint):
    """
    Machine eligibility constraint for the scheduling environment.
    This constraint defines the machines on which each task can be executed.

    Arguments:
        eligibility: Mapping[int, Iterable[int]]
            A mapping of task IDs to a list of machine IDs on which the task can be executed.

    Note:
        This constraint is limited by the setup of the scheduling environment,
        meaning that you cannot:
        - add machines that do not exist in the environment.
        - include/exclude machines that would make the task incompatible with the scheduling setup.
        - exclude all machines for a task.

        By default, if eligibility is not defined for a task, it is assumed that the task
        can be executed on the original set of machines defined by the scheduling setup.
    """

    eligibility: dict[TASK_ID, Iterable[MACHINE_ID]]

    def __init__(self, eligibility: Mapping[Int, Iterable[Int]]):
        self.eligibility = {
            TASK_ID(task): [MACHINE_ID(machine) for machine in machines]
            for task, machines in eligibility.items()
        }

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.eligibility,),
            (),
        )

    def initialize(self, state: ScheduleState) -> None:
        for task_id, machines in self.eligibility.items():
            state.tasks[task_id].set_machines(convert_to_list(machines, MACHINE_ID))

    def get_entry(self) -> str:
        return "M_j"


class ConstantProcessingTime(PassiveConstraint):
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

    processing_time: TIME

    def __init__(self, processing_time: Int = 1):
        self.processing_time = TIME(processing_time)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.processing_time,),
            (),
        )

    def initialize(self, state: ScheduleState) -> None:
        for task in state.tasks:
            for machine in task.machines:
                task.set_processing_time(machine, self.processing_time)

    def get_entry(self) -> str:
        return f"p_j={self.processing_time}"


class MachineConstraint(Constraint):
    """
    General Parallel machine constraint, differs from DisjunctiveConstraint as the disjunctive
    constraint have groups with predefined tasks, while the machine constraint defines its groups
    based on the machine assignment of the tasks.

    This is mainly a setup constraint, if the expected behavior is to constrain tasks on which
    machines they can be assigned to, use the MachineEligibilityConstraint instead.
    """

    machine_map: list[set[TASK_ID]]

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (),
            (self.machine_map,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.machine_map,) = state

    def reset(self, state: ScheduleState) -> None:
        self.machine_map = [set() for _ in range(state.n_machines)]

        for task in state.tasks:
            for machine in task.machines:
                self.machine_map[machine].add(task.task_id)

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        task = event.task

        if not task.is_fixed():
            return

        machine = task.get_assignment()
        if machine == GLOBAL_MACHINE_ID:
            return

        for m in task.machines:
            self.machine_map[m].discard(task.task_id)

        end_time = task.get_end_lb()

        for other_task in self.machine_map[machine]:
            state.tight_start_lb(other_task, end_time, machine)

        # machine_ends: dict[MACHINE_ID, TIME] = {}
        # for task in state.tasks_to_propagate:
        #     if task.is_fixed():
        #         machine = task.get_assignment()

        #         machine_ends[machine] = task.get_end_lb()

        # for machine in machine_ends:
        #     end_time = machine_ends[machine]

        #     for task in state.awaiting_tasks:
        #         if machine not in task.machines:
        #             continue

        #         state.tight_start_lb(task.task_id, end_time, machine)

    def is_complete(self, state: ScheduleState) -> bool:
        "Check if the machine constraint is complete."
        return all(len(task.machines) == state.n_machines for task in state.tasks)


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

    def __init__(self, precedence: Mapping[Int, Sequence[Int]]):
        self.precedence = {
            TASK_ID(task): [TASK_ID(child) for child in children]
            for task, children in precedence.items()
        }

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.precedence,),
            (),
        )

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[Int, Int]]) -> Self:
        """
        Create a PrecedenceConstraint from a list of edges.

        Arguments:
            edges: Iterable[tuple[int, int]]
                A list of tuples representing the edges of the precedence graph.
                Each tuple (parent, child) indicates that the parent task must be completed
                before the child task can start.
        """
        precedence: dict[Int, list[Int]] = {}

        for parent, child in edges:
            if parent not in precedence:
                precedence[parent] = []

            precedence[parent].append(child)

        return cls(precedence)

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

    def reset(self, state: ScheduleState) -> None:
        tasks = state.tasks
        for task_id in topological_sort(self.precedence, state.n_tasks):
            task = tasks[task_id]

            end_time = task.get_end_lb()

            for child_id in self.precedence[task_id]:
                state.tight_start_lb(child_id, end_time)

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        task = event.task

        if event.field.is_lower_bound() and task.task_id in self.precedence:
            end_time = task.get_end_lb()

            for child_id in self.precedence[task.task_id]:
                state.tight_start_lb(child_id, end_time)

        # to_propagate = set(task for task in state.tasks_to_propagate)

        # while to_propagate:
        #     task = to_propagate.pop()

        #     end_time = task.get_end_lb()

        #     for child_id in self.precedence.get(task.task_id, []):
        #         child = state.tasks[child_id]

        #         if child.get_start_lb() < end_time:
        #             state.tight_start_lb(child_id, end_time)
        #             to_propagate.add(child)

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

    transposed_precedence: dict[TASK_ID, list[TASK_ID]]

    def __init__(self, precedence: Mapping[Int, Sequence[Int]]):
        super().__init__(precedence)

        self.transposed_precedence = {}

        for task_id, children in self.precedence.items():
            for child_id in children:
                self.transposed_precedence.setdefault(child_id, [])
                self.transposed_precedence[child_id].append(task_id)

        if not self.is_intree():
            raise ValueError("No-wait constraint must be an in-tree.")

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        super().propagate(event, time, state)

        task = event.task

        if task.is_fixed():
            if task.task_id in self.precedence:
                end_time = task.get_end_lb()

                for child_id in self.precedence[task.task_id]:
                    state.tight_start_ub(child_id, end_time)

        elif event.field.is_lower_bound() and task.task_id in self.transposed_precedence:
            start_time = task.get_start_lb()
            for parent_id in self.transposed_precedence[task.task_id]:
                state.tight_end_lb(parent_id, start_time)

        # for task in reversed(self.tasks_order):
        #     if task.is_fixed():
        #         end_time = task.get_end_lb()

        #         for child_id in self.precedence.get(task.task_id, []):
        #             state.tight_start_lb(child_id, end_time)

        #     else:
        #         max_children_start = task.get_end_lb()
        #         for child_id in self.precedence.get(task.task_id, []):
        #             child = state.tasks[child_id]

        #             child_lb = child.get_start_lb()

        #             if max_children_start < child_lb:
        #                 max_children_start = child_lb

        #         state.tight_end_lb(task.task_id, max_children_start)

    def get_entry(self) -> str:
        return "nwt"


class DisjunctiveConstraint(Constraint):
    groups_map: list[set[TASK_ID]]

    def __init__(self, task_groups: Iterable[Iterable[Int]]):
        self.groups_map = [set(convert_to_list(task_group, TASK_ID)) for task_group in task_groups]

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.groups_map,),
            (),
        )

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        task = event.task

        if not task.is_fixed() or event.field.is_lower_bound():
            return

        for group_tasks in self.groups_map:
            if task.task_id not in group_tasks:
                continue

            end_time = task.get_end_lb()

            for other_task_id in group_tasks:
                state.tight_start_lb(other_task_id, end_time)

        # for task in state.tasks_to_propagate:
        #     for group in self.groups_map[task.task_id]:
        #         self.group_free[group] = task.get_end_lb()

        # for task in state.awaiting_tasks:
        #     for group in self.groups_map[task.task_id]:
        #         state.tight_start_lb(task.task_id, self.group_free[group])


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
    release_dates: list[TIME]

    def __init__(self, release_dates: str = "release_time"):
        self.release_tag = release_dates

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.release_tag,),
            (self.release_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.release_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.release_dates = convert_to_list(state.instance[self.release_tag], TIME)

    def reset(self, state: ScheduleState) -> None:
        for task_id, release_time in enumerate(self.release_dates):
            state.tight_start_lb(task_id, release_time)

    def get_entry(self) -> str:
        if self.release_dates:
            release_time = self.release_dates[TASK_ID(0)]

            for rt in self.release_dates:
                if rt != release_time:
                    return "r_j"

            return f"r_j={release_time}"

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
    due_dates: list[TIME]

    const_due: TIME | None

    def __init__(self, due_dates: str = "due_dates", const_due: Int | None = None):
        self.due_tag = due_dates
        self.const_due = TIME(const_due) if const_due is not None else None

        self.due_dates = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.const_due),
            (self.due_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.due_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        if self.const_due is None:
            self.due_dates = convert_to_list(state.instance[self.due_tag], TIME)

    def reset(self, state: ScheduleState) -> None:
        if self.const_due is not None:
            for task in state.tasks:
                state.tight_end_ub(task.task_id, self.const_due)

        else:
            for task_id, due_time in enumerate(self.due_dates):
                state.tight_end_ub(task_id, due_time)

    def get_entry(self) -> str:
        if self.const_due is not None:
            return f"d_j={self.const_due}"

        if self.due_dates:
            due_time = self.due_dates[0]

            for dt in self.due_dates[1:]:
                if dt != due_time:
                    return "d_j"

            return f"d_j={due_time}"

        return "d_j"


class ResourceConstraint(Constraint):
    """
    Resource constraint for the scheduling environment.

    This constraint defines the renewable resources available for tasks and their usage.
    The resources can be defined as a list of capacities and a list of resource usage for each task.

    Arguments:
        capacities: Iterable[float]
            A list of capacities for each renewable resource. The length of the list should be equal to the
            number of resources.

        resource_usage: Iterable[str]
            A list of strings that define the resource usage for each task.
            Each string refers to a column in the tasks data that contains the
            resource usage for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    resource_tags: list[str]
    capacities: list[float]
    resources: list[list[float]]

    next_available_time: list[list[TIME]]
    available_resources: list[list[float]]

    def __init__(self, capacities: Iterable[Float], resource_usage: Iterable[str]) -> None:
        self.capacities = convert_to_list(capacities, float)
        self.resource_tags = list(resource_usage)

        self.resources = []

        self.next_available_time = [[] for _ in self.resource_tags]
        self.available_resources = [[] for _ in self.resource_tags]

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.capacities, self.resource_tags),
            (self.resources, self.next_available_time, self.available_resources),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.resources, self.next_available_time, self.available_resources) = state

    def initialize(self, state: ScheduleState) -> None:
        self.resources = [
            convert_to_list(state.instance[resource], float) for resource in self.resource_tags
        ]

    def reset(self, state: ScheduleState) -> None:
        for i in range(len(self.resource_tags)):
            self.next_available_time[i] = [MAX_TIME]
            self.available_resources[i] = [self.capacities[i]]

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        task = event.task

        if not task.is_fixed():
            return

        for i, task_resources in enumerate(self.resources):
            resource_usage = task_resources[task.task_id]

            if resource_usage <= 0:
                continue

            idx = binary_search(self.available_resources[i], resource_usage, decreasing=True)

            self.next_available_time[i].insert(idx, task.get_end_lb())
            self.available_resources[i].insert(
                idx, self.available_resources[i][idx - 1] - resource_usage
            )

        for other_task in state.awaiting_tasks:
            for i, task_resources in enumerate(self.resources):
                resource_usage = task_resources[other_task.task_id]

                if resource_usage <= 0:
                    continue

                idx = binary_search(self.available_resources[i], resource_usage, decreasing=True)

                earliest_start = self.next_available_time[i][idx - 1] if idx > 0 else time

                state.tight_start_lb(other_task.task_id, earliest_start)

        # for i, available_time in enumerate(self.next_available_time):
        #     while available_time and available_time[-1] <= time:
        #         available_time.pop()
        #         self.available_resources[i].pop()

        # for task in state.tasks_to_propagate:
        #     for i, task_resources in enumerate(self.resources):
        #         resource_usage = task_resources[task.task_id]

        #         if resource_usage <= 0:
        #             continue

        #         task_end = task.get_end_lb()
        #         idx = binary_search(self.next_available_time[i], task_end, decreasing=True)

        #         self.next_available_time[i].insert(idx, task_end)
        #         self.available_resources[i].insert(
        #             idx, self.available_resources[i][idx - 1] - resource_usage
        #         )


class NonRenewableResourceConstraint(Constraint):
    """
    Resource constraint for the scheduling environment.
    This constraint defines the non-renewable resources available for tasks and their usage.
    The resources can be defined as a list of capacities and a list of resource usage for each task.

    Arguments:
        capacities: Iterable[float]
            A list of capacities for each non-renewable resource. The length of the list should
            be equal to the number of resources.

        resource_usage: Iterable[str]
            A list of strings that define the resource usage for each task.
            Each string refers to a column in the tasks data that contains the
            resource usage for each task.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    resource_tags: list[str]
    capacities: list[float]
    resources: list[list[float]]

    current_capacities: list[float]

    def __init__(self, capacities: Iterable[Float], resource_usage: Iterable[str]) -> None:
        self.capacities = convert_to_list(capacities, float)
        self.resource_tags = list(resource_usage)

        self.current_capacities = self.capacities.copy()

        self.resources = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.capacities, self.resource_tags),
            (self.resources, self.current_capacities),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.resources, self.current_capacities) = state

    def initialize(self, state: ScheduleState) -> None:
        self.resources.clear()

        for resource in self.resource_tags:
            self.resources.append(convert_to_list(state.instance[resource], float))

    def reset(self, state: ScheduleState) -> None:
        self.current_capacities = self.capacities.copy()

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        task = event.task

        if not task.is_fixed():
            return

        for i, task_resources in enumerate(self.resources):
            resource_usage = task_resources[task.task_id]

            if resource_usage <= 0:
                continue

            self.current_capacities[i] -= resource_usage

            for other_task in state.awaiting_tasks:
                other_usage = task_resources[other_task.task_id]

                if other_usage <= 0:
                    continue

                if self.current_capacities[i] < other_usage:
                    state.set_infeasible(other_task.task_id)


SetupTimes: TypeAlias = Mapping[Int, Mapping[Int, Int]] | Callable[[int, int, Any], Int]


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

    def __init__(self, setup_times: SetupTimes) -> None:
        if callable(setup_times):
            self.setup_fn = setup_times
            self.original_setup_times = {}

        else:
            self.original_setup_times = {
                TASK_ID(task): {TASK_ID(child): TIME(time) for child, time in children.items()}
                for task, children in setup_times.items()
            }

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.setup_fn or self.original_setup_times,),
            (self.original_setup_times,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.original_setup_times,) = state

    def initialize(self, state: ScheduleState) -> None:
        if self.setup_fn is not None:
            self.original_setup_times.clear()

            for task_id in range(state.n_tasks):
                self.original_setup_times[TASK_ID(task_id)] = {}

                for child_id in range(state.n_tasks):
                    if task_id == child_id:
                        continue

                    setup_time = TIME(self.setup_fn(task_id, child_id, state))

                    if setup_time > 0:
                        self.original_setup_times[TASK_ID(task_id)][TASK_ID(child_id)] = setup_time

    def reset(self, state: ScheduleState) -> None:
        self.setup_times = {
            task_id: children.copy() for task_id, children in self.original_setup_times.items()
        }

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        task = event.task

        if not task.is_fixed() or task.task_id not in self.setup_times:
            return

        end_time = task.get_end_lb()

        for child_id in list(self.setup_times[task.task_id].keys()):
            child = state.tasks[child_id]

            if child.is_fixed():
                self.setup_times[task.task_id].pop(child_id)
                continue

            setup_time = self.setup_times[task.task_id][child_id]

            state.tight_start_lb(child_id, end_time + setup_time)

        # for task_id in list(self.setup_times.keys()):
        #     task = state.tasks[task_id]

        #     if task.is_fixed():
        #         self.setup_times.pop(task_id)
        #         continue

        #     if not task.is_fixed():
        #         continue

        #     children = self.setup_times[task_id]

        #     for child_id, setup_time in children.items():
        #         child = state.tasks[child_id]

        #         if child.is_fixed():
        #             continue

        #         block_end = task.get_end_lb() + setup_time

        #         state.tight_start_lb(child_id, block_end)


class MachineBreakdownConstraint(Constraint):
    """
    Machine breakdown constraint for the scheduling environment.

    This constraint defines the breakdowns for machines, which are the times
    when the machines are unavailable for task execution. The breakdowns can be defined
    as a mapping of machine IDs to a list of (start_time, end_time) tuples representing
    the breakdown intervals.

    Arguments:
        breakdowns: Mapping[int, Iterable[tuple[int, int]]]
            A mapping of machine IDs to a list of (start_time, end_time) tuples
            representing the breakdown intervals for each machine.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    breakdowns: dict[MACHINE_ID, list[tuple[TIME, TIME]]]
    next_breakdown: dict[MACHINE_ID, int]

    def __init__(self, breakdowns: Mapping[Int, Iterable[tuple[Int, Int]]]):
        self.breakdowns = {
            MACHINE_ID(machine): sorted((TIME(start), TIME(end)) for start, end in intervals)
            for machine, intervals in breakdowns.items()
        }

        self.next_breakdown = {machine: 0 for machine in self.breakdowns}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.breakdowns,),
            (self.next_breakdown,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.next_breakdown,) = state

    @classmethod
    def from_machine_step_function(cls, step_function: Mapping[Int, Int]) -> Self:
        """
        Create a MachineBreakdownConstraint from a machine step function.
        This function is only useful in Parallel Machine Environments, where machines are
        indistinguishable and if suffices to define the number of available machines at each
        time step.

        Arguments:
            step_function: Mapping[int, int]
                A mapping of time steps to the number of available machines from that time onward.

            name: Optional[str] = None
                An optional name for the constraint.
        """
        breakdowns: dict[Int, list[tuple[Int, Int]]] = {}

        sorted_times = sorted([int(time) for time in step_function.keys()])
        n_machines = max(int(count) for count in step_function.values())

        for time in sorted_times:
            available_machines = int(step_function[TIME(time)])

            for machine in range(available_machines):
                if machine not in breakdowns:
                    continue

                start, end = breakdowns[machine][-1]
                if end == MAX_TIME:
                    breakdowns[machine][-1] = (start, time)

            for machine in range(available_machines, n_machines):
                breakdowns.setdefault(machine, [])

                if breakdowns[machine] and breakdowns[machine][-1][1] == time:
                    continue

                breakdowns[machine].append((time, MAX_TIME))

        return cls(breakdowns)

    def reset(self, state: ScheduleState) -> None:
        for machine in self.breakdowns:
            self.next_breakdown[machine] = 0

    def propagate(self, event: Event, time: TIME, state: ScheduleState) -> None:
        task = event.task

        if task.is_fixed():
            return

        for machine in task.machines:
            if machine not in self.breakdowns:
                continue

            current_index = self.next_breakdown[machine]
            breakdown_intervals = self.breakdowns[machine]

            end_lb = task.get_end_lb(machine)
            start_lb = task.get_start_lb(machine)

            while (
                current_index < len(breakdown_intervals)
                and breakdown_intervals[current_index][1] <= time
            ):
                self.next_breakdown[machine] += 1
                current_index += 1

            while current_index < len(breakdown_intervals):
                start, end = breakdown_intervals[current_index]

                if end_lb <= start:
                    break

                if start_lb < end:
                    state.tight_start_lb(task.task_id, end, machine)
                    break

        # for machine in self.breakdowns:
        #     current_index = self.next_breakdown[machine]
        #     breakdown_intervals = self.breakdowns[machine]

        #     while current_index < len(breakdown_intervals):
        #         start, end = breakdown_intervals[current_index]

        #         if time < end:
        #             break

        #         current_index += 1

        #     self.next_breakdown[machine] = current_index

        # for task in state.awaiting_tasks:
        #     for machine in task.machines:
        #         if machine not in self.breakdowns:
        #             continue

        #         current_index = self.next_breakdown[machine]
        #         breakdown_intervals = self.breakdowns[machine]

        #         end_lb = task.get_end_lb(machine)
        #         start_lb = task.get_start_lb(machine)

        #         while current_index < len(breakdown_intervals):
        #             start, end = breakdown_intervals[current_index]

        #             if end_lb <= start or start_lb >= end:
        #                 break

        #             state.tight_start_lb(task.task_id, end, machine)

        #             end_lb = task.get_end_lb(machine)
        #             start_lb = end
        #             current_index += 1

    def get_entry(self) -> str:
        return "brkdwn"


# ------------------------------------------------------------------------------------------------
#                                   Future implementations
# ------------------------------------------------------------------------------------------------


class BatchConstraint(Constraint):
    def __post_init__(self) -> None:
        raise NotImplementedError("BatchConstraint is not implemented yet.")

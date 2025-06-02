from typing import Any, Mapping, Iterable, TypeVar, Optional, Self
from copy import deepcopy

from abc import ABC
from textwrap import dedent
import re

from .tasks import Tasks, Status
from .utils import convert_to_list, topological_sort, binary_search, is_iterable_type, scale_to_int

class Constraint(ABC):
    """
    Base class for all constraints in the scheduling environment.
    This class provides a common interface for any piece in the scheduling environment that
    interacts with the tasks by limiting when they can be executed, how they are assigned to
    machines, etc.
    """
    name: str = ""
    tags: dict[str, str]

    def __init__(self, name: Optional[str] = None) -> None:
        if name is not None and not re.match(r'^[a-zA-Z0-9_]+$', name):
            raise ValueError(
                "Constraint name must be alphanumeric and cannot contain spaces or special characters."
            )

        self.name = name if name else self.__class__.__name__
        self.tags = {}

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
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.precedence = {
            task: convert_to_list(tasks) for task, tasks in precedence.items()
        }

        self.no_wait = no_wait
        self.original_precedence = deepcopy(self.precedence)

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[int, int]], no_wait: bool = False, name: Optional[str] = None) -> Self:
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
            % (Constraint) Precedence constraint {"no wait" if self.no_wait else ""}
            int: num_edges_{self.name};

            array[1..num_edges_{self.name}, 1..2] of 1..num_tasks: edges_{self.name};

            constraint forall(i in 1..num_edges_{self.name}) (
                end[edges_{self.name}[i, 1], num_parts] {operator} start[edges_{self.name}[i, 2], 1]
            );
        """

        return dedent(model)

    def export_data(self) -> str:
        data = f"edges_{self.name} = [|\n"

        num_edges = 0
        for task, children in self.original_precedence.items():
            for child in children:
                data += f"{task+1}, {child+1} |\n"
                num_edges += 1

        data += "|];"

        return f"num_edges_{self.name} = {num_edges};\n{data}"

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
    def __init__(self, precedence: Mapping[int, Iterable[int]], name: Optional[str] = None,):
        super().__init__(precedence, no_wait=True, name=name)


_T = TypeVar("_T")
class DisjunctiveConstraint(Constraint):
    original_disjunctive_groups: dict[Any, list[int]]

    def __init__(
        self,
        disjunctive_groups: Mapping[_T, Iterable[int]] | str,
        name: Optional[str] = None
    ):
        super().__init__(name)

        if isinstance(disjunctive_groups, str):
            self.original_disjunctive_groups = {}
            self.tags['disjunctive_groups'] = disjunctive_groups

        else:
            self.original_disjunctive_groups = {
                group: convert_to_list(tasks) for group, tasks in disjunctive_groups.items()
            }

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'disjunctive_groups' in self.tags:
            for task_id in range(len(tasks)):
                group = tasks.data[self.tags['disjunctive_groups']][task_id]

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

    def export_model(self) -> str:
        model = f"""\
            % (Constraint) Disjunctive constraint
            int: num_groups_{self.name};
            array[1..num_groups_{self.name}] of set of int: group_tasks_{self.name};

            constraint forall(group in group_tasks_{self.name}) (
                disjunctive([start[t, p] | p in 1..num_parts, t in group],
                            [duration[t, p] | p in 1..num_parts, t in group])
            );
        """

        return dedent(model)

    def export_data(self) -> str:
        data = f"num_groups_{self.name} = {len(self.original_disjunctive_groups)};\n"
        data += f"group_tasks_{self.name} = [\n"
        for tasks in self.original_disjunctive_groups.values():
            data += "    {" + ', '.join([
                str(task_id + 1) for task_id in tasks
            ]) + "},\n"
        data += "];\n"

        return dedent(data)


class ReleaseDateConstraint(Constraint):
    release_dates: dict[int, int]

    def __init__(
            self,
            release_dates: Mapping[int, int] | str = 'release_time',
            name: Optional[str] = None
        ):
        super().__init__(name)

        if isinstance(release_dates, str):
            self.release_dates = {}
            self.tag = release_dates

        else:
            self.release_dates = {task: date for task, date in release_dates.items()}

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

    def __init__(
            self,
            deadlines: Mapping[int, int] | str = 'due_dates',
            name: Optional[str] = None
        ):
        super().__init__(name)

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
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)

        self.capacities = convert_to_list(capacities)

        if is_iterable_type(resource_usage, str):
            self.resources = [
                {} for _ in range(len(self.capacities))
            ]
            self.tags = {resouce_name: "" for resouce_name in resource_usage}

        else:
            assert is_iterable_type(resource_usage, dict)
            self.resources = [
                {task: usage for task, usage in resources.items()}
                for resources in resource_usage
            ]

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        for i, resouce_name in enumerate(self.tags):
            resource_feature = convert_to_list(tasks.data[resouce_name], float)

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
    
    def export_model(self) -> str:
        return dedent(f"""\
            % (Constraint) Resource constraint
            int: num_resources_{self.name};
            array[1..num_resources_{self.name}] of int: capacities_{self.name}

            array[1..num_resources_{self.name}, 1..num_tasks] of int: resources_{self.name};

            constraint forall(r in 1..num_resources_{self.name})(
                cumulative(
                    [start[t, p] | t in 1..num_tasks, p in 1..num_parts where resources_{self.name}[r, t] > 0],
                    [duration[t, p] | t in 1..num_tasks, p in 1..num_parts where resources_{self.name}[r, t] > 0],
                    [resources_{self.name}[r, t] | t in 1..num_tasks, p in 1..num_parts where resources_{self.name}[r, t] > 0],
                    capacities_{self.name}[r]
                )
            );
        """)

    def export_data(self) -> str:
        new_line = '\n'

        resources_str = f"resources_{self.name} = [|\n"
        capacities_str = f"capacities_{self.name} = ["

        for i, resource in enumerate(self.resources):
            row = list(resource.values()) + [self.capacities[i]]
            int_row = scale_to_int(row)

            resources_str += ', '.join(map(str, int_row[:-1]))+ " |"
            capacities_str += str(int_row[-1])

            if i == len(self.resources) - 1:
                resources_str += "];\n"
                capacities_str += "];\n"

            else:
                resources_str += new_line
                capacities_str += ", "

        return f"num_resources_{self.name} = {len(self.resources)};\n" + \
                capacities_str + \
                resources_str

class MachineConstraint(Constraint):
    """
    General Parallel machine constraint, differs from DisjunctiveConstraint as the disjunctive
    constraint have groups with predefined tasks, while the machine constraint defines its groups
    based on the machine assignment of the tasks.

    Arguments:
        machine_constraint: A list of lists of machine ids, each sublist representing the set of
            machines that the corresponding task can be assigned to. The length of the outer list
            should be equal to the number of tasks. Finally, if None is provided, then every task can be
            processed on every machine.
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
            self.machine_constraint = []
            self.tags['machine'] = machine_constraint

        elif machine_constraint is None:
            self.machine_constraint = []
            self.complete = True

        else:
            assert is_iterable_type(machine_constraint, list)

            self.machine_constraint = [
                convert_to_list(tasks) for tasks in machine_constraint
            ]

        # Time when the machine is going to be freed
        self.machine_free: dict[int, int] = {}

    @property
    def max_M_j(self) -> int:
        if self.complete:
            return max(len(task.start_bounds) for task in self.tasks)

        return max(len(machines) for machines in self.machine_constraint)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'machine' in self.tags:
            self.machine_constraint = [
                [int(machine)] for machine in tasks.data[self.tags['machine']]
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

    def export_model(self) -> str:
        if self.complete or self.max_M_j == 1:
            return ""

        return dedent(f"""\
            % (Constraint) Machine Constraint (M_j)
            array[1..num_tasks] of set of 1..num_machines: possible_machines_{self.name};
                            
            constraint forall(t in 1..num_tasks, p in 1..num_parts) (
                assignment[t, p] in possible_machines_{self.name}[t]
        """)

class SetupConstraint(Constraint):
    """
    Setup constraint for the scheduling environment.
    This constraint is used to define the setup time between tasks.
    """
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
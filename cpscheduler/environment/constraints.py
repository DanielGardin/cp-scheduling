from typing import Any, Self, Mapping, Sequence, ClassVar, Iterable, Optional
from copy import deepcopy

from .variables import IntervalVars
from. utils import MAX_INT, AVAILABLE_SOLVERS, convert_to_list, topological_sort, is_iterable_type, binary_search

class Constraint:
    is_parameterized: ClassVar[bool] = False

    def export_constraint(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        return ""


    def propagate(self, time: int) -> None:
        """
            Given a state of the tasks, propagate the constraint to ensure the constraint is not violated.
        """
        ...

    def reset(self) -> None:
        """
            Reset the constraint to its original state and ensure the constraint is not violated
            in the initial state of the tasks.
        """
        ...


    def set_parameters(self, *args: Any, **kwargs: Any) -> None:
        """
        If the constraint is parameterized, the parameters can be set to change the behavior of the constraint.
        """
        ...


class PrecedenceConstraint(Constraint):
    tasks: IntervalVars
    precedence_map: dict[int, list[int]]
    in_degree: list[int]
    out_degree: list[int]
    sorted_tasks: list[int]
    to_update: bool

    def __init__(self, interval_var: IntervalVars, precedence_map: Mapping[int, Sequence[int]]) -> None:
        """
            Add a precedence constraint to the tasks.

            Parameters:
            ----------
            interval_var: IntervalVars
                The interval variables representing the tasks.
            precedence_map: list[NDArray[np.int32]]
                A list with n_tasks elements, each one with . Must be a Directed Acyclic Graph (DAG) over the tasks.
        """
        self.precedence_map = {task: list(adjecent) for task, adjecent in precedence_map.items()}
        self.original_precedence_map = deepcopy(self.precedence_map)

        self.tasks = interval_var
        self.to_update = True


    @classmethod
    def jobshop_precedence(cls, tasks: IntervalVars, job_feature: str, operation_feature: str) -> Self:
        jobs: list[Any]    = tasks[job_feature]
        operations: list[int] = tasks[operation_feature]

        precedence_map: dict[int, list[int]] = {task: [] for task in range(len(tasks))}

        tasks_order: dict[Any, list[int]] = {}

        for task in range(len(tasks)):
            job = jobs[task]
            op  = operations[task]

            if job not in tasks_order:
                tasks_order[job] = []

            if op >= len(tasks_order[job]):
                tasks_order[job].extend([-1] * (op - len(tasks_order[job]) + 1))

            tasks_order[job][op] = task

        for job, task_order in tasks_order.items():
            for i in range(1, len(task_order)):
                precedence_map[task_order[i-1]].append(task_order[i])

        return cls(tasks, precedence_map)


    def reset(self) -> None:
        self.precedence_map = deepcopy(self.original_precedence_map)
        self.in_degree  = [0] * len(self.tasks)
        self.out_degree = [len(self.precedence_map.get(task, [])) for task in range(len(self.tasks))]

        for adjecent in self.precedence_map.values():
            for next_task in adjecent:
                self.in_degree[next_task] += 1

        self.to_update = True
        self.propagate(0)



    # Untested feature, use with caution.
    def add_precedence(self, task1: int, task2: int) -> None:
        if task2 not in self.precedence_map[task1]:
            self.precedence_map[task1].append(task2)

            self.out_degree[task1] += 1
            self.in_degree[task2]  += 1

            if self.sorted_tasks:
                pos1 = self.sorted_tasks.index(task1) if task1 in self.sorted_tasks else len(self.sorted_tasks)
                pos2 = self.sorted_tasks.index(task2) if task2 in self.sorted_tasks else MAX_INT

                if pos1 > pos2:
                    self.to_update = True



    def remove_precedence(self, task1: int, task2: int) -> None:
        if task2 in self.precedence_map[task1]:
            self.precedence_map[task1].remove(task2)

            self.out_degree[task1] -= 1
            self.in_degree[task2]  -= 1

            # We do not need to propagate leaf nodes further.
            if self.sorted_tasks and self.out_degree[task1] == 0:
                self.sorted_tasks.remove(task1)


    def export_constraint(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        if solver == 'cplex':
            constraints = [
                f"endBeforeStart({self.tasks.get_var_name(task)}, {self.tasks.get_var_name(next_task)});"
                for task in range(len(self.tasks)) for next_task in self.precedence_map.get(task, [])
            ]

        else:
            constraints = [
                f"model.Add({self.tasks.get_var_name(task)}_end <= {self.tasks.get_var_name(next_task)}_start)"
                for task in range(len(self.tasks)) for next_task in self.precedence_map.get(task, [])
            ]

        return "\n".join(constraints)


    # TODO: Add support for propagating only necessary tasks. Use some Sat-CP algorithm.
    # TODO: Implement a depth limit optimization as we do not need to propagate constraint too far
    # into the future.
    def propagate(self, time: int) -> None:
        if self.to_update:
            self.sorted_tasks = topological_sort(self.precedence_map, self.in_degree)
            self.to_update    = False

        i = 0
        while True:
            task = self.sorted_tasks[i]

            if self.in_degree[task] == 0 and not self.tasks.is_fixed(task) and self.tasks.get_start_lb(task) < time:
                self.tasks.set_start_lb(task, time)

            end_time = max(self.tasks.get_end_lb(task), time)

            next_tasks = self.precedence_map[task]

            for next_task in next_tasks:
                if self.tasks.is_fixed(next_task):
                    self.remove_precedence(task, next_task)
                    continue

                self.tasks.set_start_lb(next_task, end_time)

            # If the task is removed from the previous loop, we do not need to increment the counter.
            if i >= len(self.sorted_tasks) - 1:
                break

            if self.sorted_tasks[i] == task:
                i += 1



class NonOverlapConstraint(Constraint):
    NAME = "NonOverlap_$1"

    def __init__(self, interval_var: IntervalVars, non_overlaping_groups: Mapping[str, Sequence[int]]):
        """
            Add a non-overlapping constraint to the tasks.

            Parameters:
            ----------
            interval_var: IntervalVars
                The interval variables representing the tasks.
            non_overlaping_groups: NDArray[np.bool], shape=(n_tasks, n_groups)
                The non-overlapping matrix. Each row represents a task and each column a group.
        """
        self.tasks = interval_var

        self.non_overlaping_groups = {
            group_id: list(indices) for group_id, indices in non_overlaping_groups.items()
        }

    @property
    def n_groups(self) -> int:
        return len(self.non_overlaping_groups)


    @classmethod
    def jobshop_non_overlap(cls, tasks: IntervalVars, machine_feature: str) -> Self:
        machines = convert_to_list(tasks[machine_feature], str)
        non_overlaping_groups: dict[str, list[int]] = {}

        for task in range(len(tasks)):
            machine = machines[task]

            if machine not in non_overlaping_groups:
                non_overlaping_groups[machine] = []

            non_overlaping_groups[machine].append(task)

        return cls(tasks, non_overlaping_groups)


    def add_group(self, tasks: Iterable[int], group_id: Optional[str] = None) -> None:
        if group_id is None:
            group_id = str(len(self.non_overlaping_groups))

        self.non_overlaping_groups[group_id] = list(tasks)


    def remove_group(self, group_id: str) -> None:
        del self.non_overlaping_groups[group_id]


    def export_constraint(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        if solver == 'cplex':
            constraints = [
                f"{self.NAME.replace('$1', str(group_id))} = sequenceVar([{', '.join(self.tasks.get_var_name(tasks))}]);\n"
                f"noOverlap({self.NAME.replace('$1', str(group_id))});"
                for group_id, tasks in self.non_overlaping_groups.items()
            ]

        else:
            constraints = [
                f"model.AddNoOverlap([{', '.join(self.tasks.get_var_name(tasks))}])"
                for tasks in self.non_overlaping_groups.values()
            ]

        return "\n".join(constraints)


    # TODO: Add support for fixing tasks without the supposition of fixed tasks being in the past.
    def propagate(self, time: int) -> None:
        for tasks in self.non_overlaping_groups.values():
            executing_tasks = [task for task in tasks if self.tasks.is_executing(task, time)]
            max_end     = max(self.tasks.get_end_lb(executing_tasks), default=time)

            for task in tasks:
                self.tasks.set_start_lb(task, max_end)



class ReleaseTimesConstraint(Constraint):
    def __init__(self, interval_var: IntervalVars, release_times: str | Iterable[int]):
        """
            Add a release time constraint to the tasks. The release times are the earliest time
            at which a task can start. Zero indicates there is no release time.

            Parameters:
            ----------
            interval_var: IntervalVars
                The interval variables representing the tasks.
            release_times: NDArray[np.int32], shape=(n_tasks,)
                The release times for each task.
        """
        self.tasks = interval_var

        if isinstance(release_times, str):
            release_times = self.tasks[release_times]


        self.release_times = convert_to_list(release_times, int)


    def reset(self) -> None:
        for task, release_time in enumerate(self.release_times):
            if release_time > 0:
                self.tasks.set_start_lb(task, release_time)


    def set_release_time(self, task: int, release_time: int) -> None:
        self.release_times[task] = release_time



class DueDatesConstraint(Constraint):
    def __init__(self, interval_var: IntervalVars, due_dates: str | Iterable[int]):
        """
            Add a due date constraint to the tasks. The due dates are the latest time at which
            a task can end. Zero indicates there is no due date. Notice that the due date constraint
            is a hard constraint, which can lead to infeasible problems.
            Alternatively, one may option for a relaxed approach by adding a penalty to the objective
            function instead.

            Parameters:
            ----------
            interval_var: IntervalVars
                The interval variables representing the tasks.
            due_dates: NDArray[np.int32], shape=(n_tasks,)
                The due dates for each task.
        """
        self.tasks = interval_var

        if isinstance(due_dates, str):
            due_dates = self.tasks[due_dates]

        self.due_dates = convert_to_list(due_dates, int)


    def reset(self) -> None:
        for task, due_date in enumerate(self.due_dates):
            if due_date > 0:
                self.tasks.set_start_lb(task, due_date)


    def set_due_date(self, task: int, due_date: int) -> None:
        self.due_dates[task] = due_date


class ResourceCapacityConstraint(Constraint):
    is_parameterized: ClassVar[bool] = True

    def __init__(
            self,
            interval_var: IntervalVars,
            resources: str | Iterable[str] | Iterable[float] | Iterable[Iterable[float]],
            resource_capacity: Iterable[float],
        ):
        """
            Add a resource capacity constraint to the tasks. The resource capacity is the maximum
            amount of resource that can be used at the same time.

            Parameters:
            ----------
            interval_var: IntervalVars
                The interval variables representing the tasks.
            resource_taken: ArrayLike, shape=(n_tasks, n_resources)
                The resource capacity for each task.

            resource_capacity: ArrayLike, shape=(n_resources,)
                The maximum amount of resource that can be used at the same time.
        """
        self.tasks = interval_var

        if isinstance(resources, str):
            resources_matrix = [convert_to_list(interval_var[resources], float)]

        elif is_iterable_type(resources, str):
            resources_matrix = [convert_to_list(interval_var[resource], float) for resource in resources]

        elif is_iterable_type(resources, float):
            resources_matrix = [convert_to_list(resources, float)]

        else:
            assert is_iterable_type(resources, Iterable[float]) # type: ignore
            resources_matrix = [convert_to_list(resource, float) for resource in resources]

        self.resource_taken    = resources_matrix
        self.resource_capacity = convert_to_list(resource_capacity, float)


    def set_parameters(self, resource_capacity: Iterable[float]) -> None:
        for i, capacity in enumerate(resource_capacity):
            self.resource_capacity[i] = capacity


    @staticmethod
    def _available_resources(current_resources: list[float], capacity: float) -> list[float]:
        available_resources = current_resources.copy()

        available_resources[-1] = capacity - current_resources[-1]

        for i in range(len(current_resources) - 1, 0, -1):
            available_resources[i-1] = available_resources[i] - current_resources[i]

        return available_resources


    def propagate(self, time: int) -> None:
        for resource_taken, resource_capacity in zip(self.resource_taken, self.resource_capacity):
            tasks = [task for task in range(len(self.tasks)) if resource_taken[task] > 0]

            executing_tasks = [task for task in tasks if self.tasks.is_executing(task, time)]

            sorted_liberation = sorted([(self.tasks.get_end_lb(task), task) for task in executing_tasks])

            resources = self._available_resources(
                [resource_taken[task] for _, task in sorted_liberation],
                resource_capacity
            )

            for task in tasks:
                if self.tasks.is_fixed(task): continue

                index = binary_search(resources, resource_taken[task])

                if index == 0:
                    earliest_start = time
                
                else:
                    earliest_start = sorted_liberation[index-1][0]

                self.tasks.set_start_lb(task, earliest_start)


    def export_constraint(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        if solver == 'cplex':
            pulses = [
                [f"pulse({self.tasks.get_var_name(task)}, {resource[task]})"
                for task in range(len(self.tasks)) if resource[task] > 0] for resource in self.resource_taken
            ]

            resource_constraints = [
                f"resource_{i}_use = {' + '.join(pulse)};"
                for i, pulse in enumerate(pulses)
            ]

            capacity_constraints = [
                f"resource_{i}_use <= {capacity};"
                for i, capacity in enumerate(self.resource_capacity)
            ]

            return "\n".join(resource_constraints + capacity_constraints)

        else:
            tasks = [
                [self.tasks.get_var_name(task) for task in range(len(self.tasks)) if resource[task] > 0]
                for resource in self.resource_taken
            ]

            resources = [
                [resource[task] for task in range(len(self.tasks)) if resource[task] > 0]
                for resource in self.resource_taken
            ]

            capacity_constraints = [
                f"model.AddCumulative([{', '.join(tasks[i])}], {resources[i]}, {self.resource_capacity[i]})"
                for i in range(len(self.resource_taken)) 
            ]

            return "\n".join(capacity_constraints)
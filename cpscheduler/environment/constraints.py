from typing import Any, Self, Optional, Iterable, Literal, Mapping, Sequence, ClassVar
from numpy.typing import NDArray, ArrayLike

from collections import deque
from copy import deepcopy

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured 

from .variables import IntervalVars, Scalar, MAX_INT, AVAILABLE_SOLVERS


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
    def __init__(self, interval_var: IntervalVars, precedence_list: Sequence[Sequence[int]]) -> None:
        """
            Add a precedence constraint to the tasks.

            Parameters:
            ----------
            interval_var: IntervalVars
                The interval variables representing the tasks.
            precedence_list: list[NDArray[np.int32]]
                A list with n_tasks elements, each one with . Must be a Directed Acyclic Graph (DAG) over the tasks.
        """
        self.precedence_list = [list(adjecent) for adjecent in precedence_list]
        self.original_precedence_list = deepcopy(self.precedence_list)

        self.tasks = interval_var

        self.in_degree  = np.zeros(len(self.tasks), dtype=np.int32)
        self.out_degree = np.zeros(len(self.tasks), dtype=np.int32)

        self.sorted_tasks: list[int] = []


    @classmethod
    def from_precedence_matrix(cls, interval_var: IntervalVars, precedence_matrix: NDArray[np.bool]) -> Self:
        precedence_list = [[i for i in np.where(row)[0].tolist()] for row in precedence_matrix]

        return cls(interval_var, precedence_list)


    @classmethod
    def jobshop_precedence(cls, tasks: IntervalVars, job_feature: str, operation_feature: str) -> Self:
        jobs       = tasks[job_feature]
        operations = tasks[operation_feature]

        precedence_list: list[list[int]] = [[] for _ in range(len(tasks))]

        for job in np.unique(jobs):
            job_indices = np.where(jobs == job)[0]

            operations_order = np.argsort(operations[job_indices])
            job_indices = job_indices[operations_order]

            for i in range(len(job_indices) - 1):
                precedence_list[job_indices[i]].append(int(job_indices[i+1]))

        return cls(tasks, precedence_list)


    def reset(self) -> None:
        self.precedence_list = deepcopy(self.original_precedence_list)

        self.in_degree[:]  = 0
        self.out_degree[:] = 0
        for i, adjecent in enumerate(self.precedence_list):
            self.out_degree[i] = len(adjecent)
            self.in_degree[[*adjecent]] += 1

        self.sorted_tasks = self._topological_sort()

        self.propagate(0)


    # Untested feature, use with caution.
    def add_precedence(self, task1: Scalar, task2: Scalar, operation: Literal[">", "<", ">>", "<<"]) -> None:
        if len(operation) == 2:
            raise NotImplementedError("No support for direct precedence yet.")

        index1: int
        index2: int

        index1, index2 = self.tasks.get_indices([task1, task2])

        if operation == '<':
            index1, index2 = index2, index1

        if index2 not in self.precedence_list[index1]:
            self.precedence_list[index1].append(index2)

            self.out_degree[index1] += 1
            self.in_degree[index2]  += 1

            if self.sorted_tasks:
                pos1 = self.sorted_tasks.index(index1) if index1 in self.sorted_tasks else len(self.sorted_tasks)
                pos2 = self.sorted_tasks.index(index2) if index2 in self.sorted_tasks else MAX_INT

                if pos1 > pos2:
                    self.sorted_tasks = self._topological_sort()



    def remove_precedence(self, task1: Scalar, task2: Scalar) -> None:
        index1: int
        index2: int

        index1, index2 = self.tasks.get_indices([task1, task2]).tolist()

        if index2 in self.precedence_list[index1]:
            self.precedence_list[index1].remove(index2)

            self.out_degree[index1] -= 1
            self.in_degree[index2]  -= 1
            # We do not need to propagate leaf nodes further.
            if self.sorted_tasks and self.out_degree[index1] == 0:
                self.sorted_tasks.remove(index1)


    def export_constraint(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        names = self.tasks.names

        if solver == 'cplex':
            constraints = [
                f"endBeforeStart({names[i]}, {names[j]});"
                for i in range(len(self.tasks)) for j in self.precedence_list[i]
            ]

        else:
            constraints = [
                f"model.Add({names[i]}_end <= {names[j]}_start)"
                for i in range(len(self.tasks)) for j in self.precedence_list[i]
            ]

        return "\n".join(constraints)


    def _topological_sort(self) -> list[int]:
        in_degree = self.in_degree.copy()

        vertices = [i for i, degree in enumerate(in_degree) if degree == 0]

        queue: deque[int] = deque(vertices)

        topological_order = []

        while queue:
            vertex = queue.popleft()

            children = self.precedence_list[vertex]

            if children:
                topological_order.append(vertex)

            for child in children:
                in_degree[child] -= 1

                if in_degree[child] == 0:
                    queue.append(child)
        return topological_order


    # TODO: Add support for propagating only necessary tasks. Use some Sat-CP algorithm.
    # TODO: Implement a depth limit optimization as we do not need to propagate constraint too far
    # into the future.
    def propagate(self, time: int) -> None:
        is_fixed = self.tasks.is_fixed()

        for vertex in self.sorted_tasks:
            if self.in_degree[vertex] == 0 and not is_fixed[vertex] and self.tasks.start_lb[vertex] < time:
                self.tasks.to_propagate[vertex] = True
                self.tasks.start_lb[vertex] = time

            end_time = max(self.tasks.end_lb[vertex].item(), time)

            children = self.precedence_list[vertex]

            for child in children:
                if is_fixed[child]:
                    self.remove_precedence(vertex, child)
                    continue

                if self.tasks.start_lb[child] < end_time:
                    self.tasks.to_propagate[child] = True
                    self.tasks.start_lb[child] = end_time



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
        self.non_overlaping_groups = {
            group_id: list(indices) for group_id, indices in non_overlaping_groups.items()
        }

        self.tasks = interval_var


    @property
    def n_groups(self) -> int:
        return len(self.non_overlaping_groups)


    @classmethod
    def jobshop_non_overlap(cls, tasks: IntervalVars, machine_feature: str) -> Self:
        machines = tasks[machine_feature]

        non_overlaping_groups: dict[str, list[int]] = {
            str(machine): np.where(machines == machine)[0].tolist() for machine in np.unique(machines)
        }

        return cls(tasks, non_overlaping_groups)


    def add_group(self, tasks: Iterable[Scalar], group_id: Optional[str] = None) -> None:
        indices = self.tasks.get_indices(tasks)
        if group_id is None:
            group_id = f"{len(self.non_overlaping_groups)}"

        self.non_overlaping_groups[group_id] = indices.tolist()


    def remove_group(self, group_id: str) -> None:
        del self.non_overlaping_groups[group_id]


    def export_constraint(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        names = np.asarray(self.tasks.names)

        if solver == 'cplex':
            constraints = [
                f"{self.NAME.replace('$1', str(group_id))} = sequenceVar([{', '.join(names[indices])}]);\n"
                f"noOverlap({self.NAME.replace('$1', str(group_id))});"
                for group_id, indices in self.non_overlaping_groups.items()
            ]

        else:
            constraints = [
                f"model.AddNoOverlap([{', '.join(names[indices])}])"
                for indices in self.non_overlaping_groups.values()
            ]

        return "\n".join(constraints)


    # TODO: Add support for fixing tasks without the supposition of fixed tasks being in the past.
    def propagate(self, time: int) -> None:
        to_propagate = self.tasks.to_propagate
        is_fixed = self.tasks.is_fixed()

        propagate = to_propagate & is_fixed

        if not np.any(propagate):
            return

        masked_fixed_end_lb: NDArray[np.int32] = np.where(propagate, self.tasks.end_lb[:], time)

        for indices in self.non_overlaping_groups.values():
            if not np.any(propagate[indices]): continue

            group_max_end_lb = np.max(masked_fixed_end_lb[indices]).item()

            free_indices = [index for index in indices if not is_fixed[index]]

            self.tasks.to_propagate[free_indices] |= self.tasks.start_lb[free_indices] < group_max_end_lb

            self.tasks.start_lb[free_indices] = np.maximum(self.tasks.start_lb[free_indices], group_max_end_lb)


class ReleaseTimesConstraint(Constraint):
    def __init__(self, interval_var: IntervalVars, release_times: NDArray[np.int32]):
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
        self.release_times = release_times
        self.tasks = interval_var


    @classmethod
    def jobshop_release_times(cls, tasks: IntervalVars, release_time_feature: str) -> Self:
        return cls(tasks, tasks[release_time_feature])


    def reset(self) -> None:
        mask = self.release_times > 0

        self.tasks.start_lb[mask] = np.maximum(self.tasks.start_lb[mask], self.release_times[mask])


    def set_release_time(self, task: Scalar, release_time: int) -> None:
        index = self.tasks.get_indices(task)

        self.release_times[index] = release_time



class DueDatesConstraint(Constraint):
    def __init__(self, interval_var: IntervalVars, due_dates: NDArray[np.int32]):
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

        self.due_dates = due_dates
        self.tasks = interval_var


    @classmethod
    def jobshop_due_dates(cls, tasks: IntervalVars, due_date_feature: str) -> Self:
        return cls(tasks, tasks[due_date_feature])


    def reset(self) -> None:
        mask = self.due_dates > 0

        self.tasks.end_ub[mask] = np.minimum(self.tasks.end_ub[mask], self.due_dates[mask])


    def set_due_date(self, task: Scalar, due_date: int) -> None:
        index = self.tasks.get_indices(task)

        self.due_dates[index] = due_date



class ResourceCapacityConstraint(Constraint):
    is_parameterized: ClassVar[bool] = True

    def __init__(
            self,
            interval_var: IntervalVars,
            resources: ArrayLike | str | list[str],
            resource_capacity: ArrayLike,
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
        # resources: NDArray[Any]
        if isinstance(resources, str):
            resources_matrix = interval_var[resources]

        elif isinstance(resources, list):
            resources_matrix = structured_to_unstructured(interval_var[resources])

        else:
            resources_matrix = np.asarray(resources)


        self.resource_taken    = resources_matrix.reshape(len(interval_var), -1).astype(np.int32)
        self.resource_capacity = np.atleast_1d(resource_capacity)

        self.tasks = interval_var

        n_tasks = len(self.tasks)
        self.n_resources = len(self.resource_capacity)

        assert self.resource_taken.shape == (n_tasks, self.n_resources), \
            f"Resource taken shape {self.resource_taken.shape} does not match the expected shape {(n_tasks, self.n_resources)}."


    def set_parameters(self, resource_capacity: NDArray[np.int32]) -> None:
        self.resource_capacity = resource_capacity


    def propagate(self, time: int) -> None:
        to_propagate = self.tasks.to_propagate
        is_fixed = self.tasks.is_fixed()

        propagate = to_propagate & is_fixed

        if not np.any(propagate):
            return

        for resource in range(self.n_resources):
            indices = [task for task in range(len(self.tasks)) if self.resource_taken[task, resource] > 0]

            if not np.any(propagate[indices]): continue

            fixed_indices = [index for index in indices if is_fixed[index]]

            liberation_time = self.tasks.end_lb[fixed_indices]
            argsort = np.argsort(liberation_time)

            liberation_time = np.insert(
                liberation_time[argsort], 0, time
            )

            resources = self.resource_taken[fixed_indices, resource][argsort]

            cumulative_resource = np.insert(np.cumsum(resources), 0, 0.)
            available_resource = self.resource_capacity[resource] - cumulative_resource[-1] + cumulative_resource 

            free_indices  = [index for index in indices if not is_fixed[index]]

            earliest_start = liberation_time[
                np.searchsorted(available_resource, self.resource_taken[free_indices, resource])
            ]

            self.tasks.to_propagate[free_indices] |= self.tasks.start_lb[free_indices] < earliest_start
            self.tasks.start_lb[free_indices] = np.maximum(self.tasks.start_lb[free_indices], earliest_start)


    def export_constraint(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        if solver == 'cplex':
            pulses = [
                [f"pulse({self.tasks.names[task]}, {self.resource_taken[task, resource]})"
                for task in range(len(self.tasks)) if self.resource_taken[task, resource] > 0] for resource in range(self.n_resources)
            ]

            resource_constraints = [
                f"resource_{i}_use = {' + '.join(pulses[i])};"
                for i in range(self.n_resources) if pulses[i]
            ]

            capacity_constraints = [
                f"resource_{i}_use <= {self.resource_capacity[i]};"
                for i in range(self.n_resources) if pulses[i]
            ]

            return "\n".join(resource_constraints + capacity_constraints)

        else:
            resources = []
            for resource in range(self.n_resources):
                mask = self.resource_taken[:, resource] > 0
                if not np.any(mask): continue

                resource_pulses = [
                    f"{self.tasks.names[task]}"
                    for task in range(len(self.tasks)) if mask[task]
                ]
                
                resources.append(f"model.AddCumulative([{', '.join(resource_pulses)}], {self.resource_taken[mask, resource].tolist()}, {self.resource_capacity[resource]})")
            
            return "\n".join(resources)
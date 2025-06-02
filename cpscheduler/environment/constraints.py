from typing import Self, Optional, Iterable, Literal, Mapping, Sequence, TypeVar
from numpy.typing import NDArray

from collections import deque
from copy import deepcopy

from .variables import IntervalVars, Scalar

import numpy as np

class Constraint:
    def export_constraint(self) -> str:
        return ""

    def propagate(self) -> None:
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



class PrecedenceConstraint(Constraint):
    precedence_list: list[list[int]]

    def __init__(self, interval_var: IntervalVars, precedence_list: Sequence[Sequence[int]]) -> None:
        """
            Add a precedence constraint to the tasks. The precedence matrix is a boolean matrix of
            shape (n_tasks, n_tasks) where a True value at (i, j) indicates that task i must be
            completed before task j.

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


        self.propagate()



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

            self.in_degree[index2]  += 1
            self.out_degree[index1] += 1


    def remove_precedence(self, task1: Scalar, task2: Scalar) -> None:
        index1: int
        index2: int

        index1, index2 = self.tasks.get_indices([task1, task2]).tolist()

        if index2 in self.precedence_list[index1]:
            self.precedence_list[index1].remove(index2)

            self.in_degree[index2]  -= 1
            self.out_degree[index1] -= 1


    def export_constraint(self) -> str:
        names = self.tasks.names

        constraints = [
            f"endBeforeStart({names[i]}, {names[j]});"
            for i in range(len(self.tasks)) for j in self.precedence_list[i]
        ]

        return "\n".join(constraints)


    # TODO: Add support for fixing tasks without the supposition of fixed tasks being in the past.
    # e.g. fixing a task to the future without addressing previous tasks.
    def propagate(self) -> None:
        vertices: list[int] = np.where((self.in_degree == 0) & (self.out_degree > 0))[0].tolist()

        queue = deque(vertices)

        in_degree = self.in_degree.copy()

        is_fixed = self.tasks.is_fixed()

        while queue:
            vertex = queue.popleft()

            end_time = self.tasks.end_lb[vertex]

            neighbors = self.precedence_list[vertex]

            self.tasks.to_propagate[neighbors] |= self.tasks.start_lb[neighbors] < end_time

            self.tasks.set_start_bounds('lb')[neighbors] = np.maximum(self.tasks.start_lb[neighbors], end_time)

            in_degree[neighbors] -= 1

            for neighbor in neighbors:
                if is_fixed[neighbor]: self.remove_precedence(vertex, neighbor)

                if in_degree[neighbor] == 0:
                    queue.append(neighbor)


class NonOverlapConstraint(Constraint):
    non_overlaping_groups: dict[str, list[int]]

    NAME = "NonOverlap_$1"

    def __init__(self, interval_var: IntervalVars, non_overlaping_groups: Mapping[str, Sequence[int]]):
        """
            Add a non-overlapping constraint to the tasks. The non-overlapping matrix is a boolean matrix of
            shape (n_tasks, n_groups) where a True value at (i, j) indicates that a task i is in group j, and
            tasks in the same group cannot overlap.

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


    def export_constraint(self) -> str:
        names = np.asarray(self.tasks.names)

        constraints = [
            f"{self.NAME.replace('$1', str(group_id))} = sequenceVar([{', '.join(names[indices])}]);\n"
            f"noOverlap({self.NAME.replace('$1', str(group_id))});"
            for group_id, indices in self.non_overlaping_groups.items()
        ]

        return "\n".join(constraints)

    # TODO: Add support for fixing tasks without the supposition of fixed tasks being in the past.
    def propagate(self) -> None:
        to_propagate = self.tasks.to_propagate
        is_fixed = self.tasks.is_fixed()

        propagate = to_propagate & is_fixed

        if not np.any(propagate):
            return

        masked_fixed_end_lb = np.where(propagate, self.tasks.end_lb, 0)

        for indices in self.non_overlaping_groups.values():
            if not np.any(propagate[indices]): continue

            group_max_end_lb = np.max(masked_fixed_end_lb[indices])

            free_indices = [index for index in indices if not is_fixed[index]]

            self.tasks.to_propagate[free_indices] |= self.tasks.start_lb[free_indices] < group_max_end_lb

            self.tasks.set_start_bounds('lb')[free_indices] = np.maximum(self.tasks.start_lb[free_indices], group_max_end_lb)


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

        self.tasks.set_start_bounds('lb')[mask] = np.maximum(self.tasks.start_lb[mask], self.release_times[mask])


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

        self.tasks.set_end_bounds('ub')[mask] = np.minimum(self.tasks.end_ub[mask], self.due_dates[mask])


    def set_due_date(self, task: Scalar, due_date: int) -> None:
        index = self.tasks.get_indices(task)

        self.due_dates[index] = due_date



# TODO: Add support for setup times between tasks.
class SetupTimeConstraint:
    def __init__(self, interval_var: IntervalVars, setup_times: NDArray[np.int32]):
        """
            Add a setup time constraint to the tasks. The setup times are the time it takes to
            prepare the machine for a task. The setup times are added to the end time of the
            previous task and the start time of the next task. The matrix is of shape (n_tasks, n_tasks),
            where a value at (i, j) indicates the setup time after finishing task i and immediately
            starting task j.

            Parameters:
            ----------
            interval_var: IntervalVars
                The interval variables representing the tasks.
            
            setup_times: NDArray[np.int32], shape=(n_tasks, n_tasks)
                The setup times for each pair of tasks.
        """

        self.setup_times = setup_times
        self.tasks = interval_var


    @classmethod
    def jobshop_setup_times(cls, tasks: IntervalVars, setup_time_feature: str) -> Self:
        return cls(tasks, tasks[setup_time_feature])


    def set_setup_time(self, task1: Scalar, task2: Scalar, setup_time: int) -> None:
        index1, index2 = self.tasks.get_indices([task1, task2])

        self.setup_times[index1, index2] = setup_time


    def export_constraint(self) -> str:
        names = self.tasks.names

        # This is not right because setup times do not imply precedence.
        return "\n".join([
            f"endBeforeStart({names[i]}, {names[j]}, {setup_time});"
            for i, j, setup_time in zip(*np.where(self.setup_times > 0), self.setup_times[self.setup_times > 0])
        ])

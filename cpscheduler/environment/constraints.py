from typing import Self, Optional, Iterable, Literal
from numpy.typing import NDArray

from .variables import IntervalVars, Scalar

import numpy as np

class Constraint:
    def export_constraint(self) -> str:
        return NotImplemented
    
    def propagate(self) -> None:
        ...



class PrecedenceConstraint(Constraint):
    def __init__(self, interval_var: IntervalVars, precedence_matrix: NDArray[np.bool]) -> None:
        """
            Add a precedence constraint to the tasks. The precedence matrix is a boolean matrix of
            shape (n_tasks, n_tasks) where a True value at (i, j) indicates that task i must be
            completed before task j.

            Parameters:
            ----------
            interval_var: IntervalVars
                The interval variables representing the tasks.
            
            precedence_matrix: NDArray[np.bool], shape=(n_tasks, n_tasks)
                The precedence matrix. Must be a Directed Acyclic Graph (DAG) over the tasks.
        """
        self.precedence_matrix = precedence_matrix

        self.tasks = interval_var


    @classmethod
    def jobshop_precedence(cls, tasks: IntervalVars, job_feature: str, operation_feature: str) -> Self:
        jobs       = tasks[job_feature]
        operations = tasks[operation_feature]

        precedence_matrix = (operations[:, None] + 1 == operations) & (jobs[:, None] == jobs)

        return cls(tasks, precedence_matrix)


    def add_precedence(self, task1: Scalar, task2: Scalar, operation: Literal[">", "<", ">>", "<<"]) -> None:
        if len(operation) == 2:
            raise NotImplementedError("No support for direct precedence yet.")

        index1, index2 = self.tasks.get_indices([task1, task2])

        if operation == '<':
            index1, index2 = index2, index1

        self.precedence_matrix[index1, index2] = True


    def remove_precedence(self, task1: Scalar, task2: Scalar) -> None:
        index1, index2 = self.tasks.get_indices([task1, task2])

        self.precedence_matrix[index1, index2] = False


    def export_constraint(self) -> str:
        names = self.tasks.names

        constraints = [
            f"endBeforeStart({names[i]}, {names[j]});"
            for i, j in zip(*np.where(self.precedence_matrix))
        ]

        return "\n".join(constraints)


    # TODO: Add support for fixing tasks without the supposition of fixed tasks being in the past.
    # e.g. fixing a task to the future without addressing previous tasks.
    def propagate(self) -> None:
        adjecency_matrix = self.precedence_matrix.copy()

        # If a task is fixed, then t_precedent > t_fixed is secured.
        adjecency_matrix[:, self.tasks.is_fixed()] = False

        in_degree  = np.sum(adjecency_matrix, axis=0)
        out_degree = np.sum(adjecency_matrix, axis=1)

        initial_vertices = np.where((in_degree == 0) & (out_degree > 0))[0]

        queue = list(initial_vertices)

        while queue:
            vertex = queue.pop(0)

            end_time = self.tasks.end_lb[vertex]

            adjecent_mask = adjecency_matrix[vertex]
            reduce_mask   = adjecent_mask & (self.tasks.start_lb < end_time)

            self.tasks.set_start_bounds('lb')[reduce_mask] = end_time

            in_degree[adjecent_mask] -= 1

            queue.extend(np.where((in_degree == 0) & adjecent_mask)[0])



class NonOverlapConstraint(Constraint):
    NAME = "NonOverlap_$1"

    def __init__(self, interval_var: IntervalVars, non_overlaping_groups: NDArray[np.bool], group_ids: Optional[Iterable[Scalar]] = None):
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
        n_tasks, n_groups = non_overlaping_groups.shape

        if n_tasks != len(interval_var):
            raise ValueError("The number of tasks must be equal to the number of rows in the non-overlapping matrix.")

        self.non_overlaping_groups = non_overlaping_groups
        self.tasks = interval_var

        if group_ids is None:
            group_ids = [i for i in range(n_groups)]

        self.group_ids = list(group_ids)


    @property
    def n_groups(self) -> int:
        return len(self.group_ids)


    @classmethod
    def jobshop_non_overlap(cls, tasks: IntervalVars, machine_feature: str) -> Self:
        machines = tasks[machine_feature]

        non_overlaping_groups = (machines[:, None] == np.unique(machines))

        return cls(tasks, non_overlaping_groups)


    def add_group(self, tasks_: Iterable[Scalar] | NDArray[np.bool]) -> None:
        if isinstance(tasks_, np.ndarray) and tasks_.dtype == np.bool:
            indices = tasks_
        
        else:
            indices = self.tasks.get_indices(tasks_)
            

        self.non_overlaping_groups = np.concatenate([self.non_overlaping_groups, np.zeros((len(self.tasks), len(indices)), dtype=np.bool)], axis=1)

        self.non_overlaping_groups[indices, -1] = True


    def remove_group(self, group_id: Scalar) -> None:
        i = self.group_ids.index(group_id)
        self.group_ids.pop(i)

        self.non_overlaping_groups = np.delete(self.non_overlaping_groups, i, axis=1)


    def export_constraint(self) -> str:
        names = np.asarray(self.tasks.names)

        constraints = [
            f"{self.NAME.replace('$1', str(group_id))} = sequenceVar([{', '.join(names[self.non_overlaping_groups[:, i]])}]);\n"
            f"noOverlap({self.NAME.replace('$1', str(group_id))});"
            for i, group_id in enumerate(self.group_ids)
        ]

        return "\n".join(constraints)

    # TODO: Add support for fixing tasks without the supposition of fixed tasks being in the past.
    def propagate(self) -> None:
        groups = self.non_overlaping_groups.copy()

        is_fixed = self.tasks.is_fixed()

        masked_end_lb = np.where(groups & is_fixed[:, None], self.tasks.end_lb[:, None], 0)

        max_fixed_end_lb = np.max(masked_end_lb, axis=0)

        masked_end_lb = np.where(groups & ~is_fixed[:, None], max_fixed_end_lb, 0)

        max_end_lb = np.max(masked_end_lb, axis=1)

        self.tasks.set_start_bounds('lb')[:] = np.maximum(self.tasks.start_lb, max_end_lb)



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


    def set_release_time(self, task: Scalar, release_time: int) -> None:
        index = self.tasks.get_indices(task)

        self.release_times[index] = release_time


    def export_constraint(self) -> str:
        names = self.tasks.names

        return "\n".join([
            f"startOf({name}) >= {release_time};"
            for name, release_time in zip(names, self.release_times) if release_time > 0
        ])


    def propagate(self) -> None:
        mask = self.release_times > 0

        self.tasks.set_start_bounds('lb')[mask] = np.maximum(self.tasks.start_lb[mask], self.release_times[mask])


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
    

    def set_due_date(self, task: Scalar, due_date: int) -> None:
        index = self.tasks.get_indices(task)

        self.due_dates[index] = due_date
    

    def export_constraint(self) -> str:
        names = self.tasks.names

        return "\n".join([
            f"endOf({name}) <= {due_date};"
            for name, due_date in zip(names, self.due_dates) if due_date > 0
        ])


    def propagate(self) -> None:
        mask = self.due_dates > 0

        self.tasks.set_end_bounds('ub')[mask] = np.minimum(self.tasks.end_ub[mask], self.due_dates[mask])



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

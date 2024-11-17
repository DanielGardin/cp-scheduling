from typing import Any, Optional, ClassVar
from numpy.typing import NDArray

from .variables import IntervalVars, AVAILABLE_SOLVERS, MAX_INT

import numpy as np

from collections import defaultdict


class Objective:
    is_parameterized: ClassVar[bool] = False

    def export_objective(self, minimize: bool, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        """
        Export the objective function to a string that is parsable by a selected solver.

        Parameters
        ----------
        minimize : bool
            Whether the objective function should be minimized or maximized. The default can vary depending
            on the objective function.
        
        solver : 'cplex', 'ortools'
            The solver to which the objective function will be exported. The default is 'cplex'.

        """
        return ""

    def get_current(self) -> float:
        """
        Get the current value of the objective function. This is useful for checking the performance of the
        scheduling algorithm along the episode.
        """
        return NotImplemented

    def set_parameters(self, *args: Any, **kwargs: Any) -> None:
        """
        If the objective function is parameterized, the parameters can be set to change the behavior of the objective.
        """
        ...


class Makespan(Objective):
    def __init__(self, interval_var: IntervalVars):
        """
        Classic makespan objective function, which aims to minimize the time at which all tasks are completed.

        Parameters
        ----------
        interval_var : IntervalVars
            The interval variables that represent the tasks to be scheduled.        
        """

        self.tasks = interval_var


    def export_objective(self, minimize: bool = False, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        objective_type = 'minimize' if minimize else 'maximize'

        names = self.tasks.names

        if solver == 'cplex':
            ends = [f"endOf({name})" for name in names]

            return f"makespan = max([{', '.join(ends)}]);\n{objective_type}(makespan);"
        
        else:
            makespan = f'makespan = model.NewIntVar(0, {MAX_INT}, "makespan")'
            ends = [f"{name}_end" for name in names]

            return f"{makespan}\nmodel.AddMaxEquality(makespan, [{', '.join(ends)}])\nmodel.{objective_type}(makespan)"


    def get_current(self) -> int:
        return int(np.max(self.tasks.end_lb[self.tasks.is_fixed()], initial=0))



class ClientWeightedCompletionTime(Objective):
    is_parameterized = True

    def __init__(
            self,
            interval_var: IntervalVars,
            client_weights: NDArray[np.float32],
            job_client: NDArray[np.integer[Any]],
            job_feature: str | NDArray[np.integer[Any]],
        ):
        """
        Objective function that aims to minimize the weighted completion time of each client.
        Here, each job is associated with a client.

        Parameters
        ----------
        interval_var : IntervalVars
            The interval variables that represent the tasks to be scheduled.

        client_weights : NDArray[float]
            The weights of each client.
        
        job_client : NDArray[int]
            The client associated with each job.
        
        job_feature : str | NDArray[int]
            The feature of each job that will be used to calculate the completion time.
            If a string is passed, it is assumed that the feature is a column of the interval variable.

        """

        self.tasks = interval_var

        self.job_op: NDArray[np.integer[Any]] = self.tasks[job_feature] if isinstance(job_feature, str) else job_feature

        self.n_jobs    = len(job_client)
        self.n_clients = len(client_weights)

        self.client_weights = client_weights
        self.job_client     = job_client




    def export_objective(self, minimize: bool = True, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        objective_type = 'minimize' if minimize else 'maximize'

        names = self.tasks.names

        rep = ''

        ends: list[list[str]] = [[] for _ in range(self.n_jobs)]
        job: int

        for name, job in zip(names, self.job_op):
            if solver == 'cplex':
                job_end = f"endOf({name})"
            
            else:
                job_end = f"{name}_end"

            ends[job].append(job_end)


        for job in range(self.n_jobs):
            if solver == 'cplex':
                rep += f"job{job}_completion = max([{', '.join(ends[job])}]);\n"

            else:
                rep += f"job{job}_completion = model.NewIntVar(0, {MAX_INT}, f'job{job}_completion')\n"
                rep += f"model.AddMaxEquality(job{job}_completion, [{', '.join(ends[job])}])\n"


        weighted_completion_times = ' + '.join([
            f"{weight} * job{job}_completion" for job, weight in enumerate(self.client_weights[self.job_client])
        ])

        if solver == 'cplex':
            return rep + f"{objective_type}({weighted_completion_times});"
    
        else:
            return rep + f"model.{objective_type}({weighted_completion_times})"


    def get_current(self) -> float:
        is_fixed = self.tasks.is_fixed()

        max_end = [0 for _ in range(self.n_jobs)]

        for end, job in zip(self.tasks.end_lb[is_fixed], self.job_op[is_fixed]):
            max_end[job] = max(max_end[job], end)

        current_objective_value = float(np.sum(self.client_weights[self.job_client] * max_end))

        return current_objective_value


    def set_parameters(self, client_weights: NDArray[np.float32]) -> None:
        self.client_weights = client_weights
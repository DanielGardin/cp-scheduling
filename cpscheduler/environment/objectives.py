from typing import Any, ClassVar, Iterable

from .variables import IntervalVars
from .utils import AVAILABLE_SOLVERS, MAX_INT, convert_to_list


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

    def get_current(self, time: int) -> float:
        """
        Get the current value of the objective function. This is useful for checking the performance of the
        scheduling algorithm along the episode.
        """
        return 0

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

        names = self.tasks.get_var_name()

        if solver == 'cplex':
            ends = [f"endOf({name})" for name in names]

            return f"makespan = max([{', '.join(ends)}]);\n{objective_type}(makespan);"

        else:
            makespan = f'makespan = model.NewIntVar(0, {MAX_INT}, "makespan")'
            ends = [f"{name}_end" for name in names]

            return f"{makespan}\nmodel.AddMaxEquality(makespan, [{', '.join(ends)}])\nmodel.{objective_type}(makespan)"


    def get_current(self, time: int) -> int:
        fixed_tasks = [task for task in range(len(self.tasks)) if self.tasks.is_fixed(task)]

        return max(self.tasks.get_end_lb(fixed_tasks), default=0)



class WeightedCompletionTime(Objective):
    is_parameterized = True

    def __init__(
            self,
            interval_var: IntervalVars,
            job_feature: str | Iterable[Any],
            weights: Iterable[float]
        ):
        """
        Objective function that aims to minimize the weighted completion time of each client.
        Here, each job is associated with a client.

        Parameters
        ----------
        interval_var : IntervalVars
            The interval variables that represent the tasks to be scheduled.

        job_feature : str | Arraylike, shape=(n_tasks,)
            The feature of containing the job associated with each task.
            If a string is passed, it is assumed that the feature is a column of the interval variable.

        weights : NDArray[float]
            The weight of each job.

        """

        self.tasks = interval_var

        self.jobs: list[Any] = self.tasks[job_feature] if isinstance(job_feature, str) else convert_to_list(job_feature)

        self.weights = convert_to_list(weights, dtype=float)
        self.n_jobs = len(self.weights)


    def export_objective(self, minimize: bool = True, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        objective_type = 'minimize' if minimize else 'maximize'

        names = self.tasks.get_var_name()

        rep = ''

        ends: dict[Any, list[str]] = {}

        for i, name in enumerate(names):
            if solver == 'cplex':
                job_end = f"endOf({name})"

            else:
                job_end = f"{name}_end"

            job = self.jobs[i]

            if job not in ends:
                ends[job] = []

            ends[self.jobs[i]].append(job_end)


        for job in ends:
            if solver == 'cplex':
                rep += f"job{job}_completion = max([{', '.join(ends[job])}]);\n"

            else:
                rep += f"job{job}_completion = model.NewIntVar(0, {MAX_INT}, f'job{job}_completion')\n"
                rep += f"model.AddMaxEquality(job{job}_completion, [{', '.join(ends[job])}])\n"


        weighted_completion_times = ' + '.join([
            f"{weight} * job{job}_completion" for job, weight in enumerate(self.weights)
        ])

        if solver == 'cplex':
            return rep + f"{objective_type}({weighted_completion_times});"

        else:
            return rep + f"model.{objective_type}({weighted_completion_times})"


    def get_current(self, time: int) -> float:
        is_fixed = self.tasks.is_fixed()

        completion_times: dict[Any, int] = {}

        for task, end_time in enumerate(self.tasks.get_end_lb()):
            if is_fixed[task]:
                job = self.jobs[task]

                if job not in completion_times:
                    completion_times[job] = 0

                completion_times[job] = max(completion_times[job], end_time)

        objective_value = sum([weight * job_end for weight, job_end in zip(self.weights, completion_times.values())], start=0.)

        return objective_value

    def set_parameters(self, weights: Iterable[float]) -> None:
        self.weights = convert_to_list(weights, dtype=float)
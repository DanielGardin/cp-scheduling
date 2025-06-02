from numpy.typing import NDArray

from .variables import IntervalVars, AVAILABLE_SOLVERS, MAX_INT

import numpy as np


class Objective:
    def export_objective(self, minimize: bool = True, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        return NotImplemented

    def get_current(self) -> float:
        return NotImplemented


class Makespan(Objective):
    def __init__(self, interval_var: IntervalVars):
        self.tasks = interval_var


    def export_objective(self, minimize: bool = True, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
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


class TotalWeigthedCompletionTime(Objective):
    def __init__(self, interval_var: IntervalVars, weights: NDArray[np.int32]):
        self.tasks = interval_var
        self.weights = weights


    def export_objective(self, minimize: bool = True, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        objective_type = 'minimize' if minimize else 'maximize'

        if solver == 'cplex':
            weighted_ends = [f"{weight} * endOf({name})" for weight, name in zip(self.weights, self.tasks.names) if weight > 0]

            return f"weighted_makespan = {' + '.join(weighted_ends)};\n{objective_type}(weighted_makespan);"

        else:
            weighted_makespan = f'weighted_makespan = model.NewIntVar(0, {MAX_INT}, "weighted_makespan")'
            return ""

    def get_current(self) -> int:
        is_fixed = self.tasks.is_fixed()

        return int(np.sum(self.weights[is_fixed] * self.tasks.end_lb[is_fixed], initial=0))

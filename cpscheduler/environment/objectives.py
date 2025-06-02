from numpy.typing import NDArray

from .variables import IntervalVars

import numpy as np

class Objective:
    def export_objective(self) -> str:
        return NotImplemented

    def get_current(self) -> float:
        return NotImplemented


class Makespan(Objective):
    def __init__(self, interval_var: IntervalVars):
        self.tasks = interval_var


    def export_objective(self) -> str:
        ends = [f"endOf({name})" for name in self.tasks.names]

        return f"makespan = max([{', '.join(ends)}]);\nminimize(makespan);"


    def get_current(self) -> int:
        return int(np.max(self.tasks.end_lb))


class TotalWeigthedCompletionTime(Objective):
    def __init__(self, interval_var: IntervalVars, weights: NDArray[np.int32]):
        self.tasks = interval_var
        self.weights = weights


    def export_objective(self) -> str:
        weighted_ends = [f"{weight} * endOf({name})" for weight, name in zip(self.weights, self.tasks.names) if weight > 0]

        return f"weighted_makespan = {' + '.join(weighted_ends)};\nminimize(weighted_makespan);"


    def get_current(self) -> int:
        return int(np.sum(self.weights * self.tasks.end_lb))
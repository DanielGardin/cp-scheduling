from typing import ClassVar, Iterable, Literal, SupportsFloat, Optional

from textwrap import dedent

from .tasks import Tasks
from .utils import convert_to_list, scale_to_int

OptimizationDirections = Literal["min", "max"]


class Objective:
    default_direction: ClassVar[OptimizationDirections] = "min"
    tasks: Tasks

    convert_direction: ClassVar[dict[OptimizationDirections, str]] = {
        "min": "minimize",
        "max": "maximize",
    }

    def __init__(self, direction: Optional[OptimizationDirections] = None) -> None:
        direction = self.default_direction if direction is None else direction

        self.direction = self.convert_direction[direction]

    def set_tasks(self, tasks: Tasks) -> None:
        self.tasks = tasks

    def get_current(self, time: int) -> SupportsFloat:
        """
        Get the current value of the objective function. This is useful for checking the performance of the
        scheduling algorithm along the episode.
        """
        return 0

    def export_model(self) -> str:
        return "solve satisfy;"

    def export_data(self) -> str:
        return ""

    def get_entry(self) -> str:
        return ""


class Makespan(Objective):
    """
    Classic makespan objective function, which aims to minimize the time at which all tasks are completed.
    """

    def get_current(self, time: int) -> int:
        task_ends = [task.get_end() for task in self.tasks if task.is_completed(time)]

        return max(task_ends, default=0)

    def export_model(self) -> str:
        model = f"""
            var int: makespan;
            constraint makespan = max(t in 1..num_tasks)(end[t, num_parts]);

            solve {self.direction} makespan;
        """

        return dedent(model)

    def get_entry(self) -> str:
        return "C_max"


class WeightedCompletionTime(Objective):
    def __init__(
        self,
        job_weights: Iterable[float],
        direction: Optional[OptimizationDirections] = None,
        scale_integer: bool = False,
    ):
        super().__init__(direction)
        self.job_weights = convert_to_list(job_weights)
        self.scale_integer = scale_integer

    def get_current(self, time: int) -> float:
        current_value = 0.0

        for job, job_weight in enumerate(self.job_weights):
            tasks = self.tasks.get_job_tasks(job)

            job_completion = max(
                [task.get_end() for task in tasks if task.is_completed(time)], default=0
            )

            current_value += job_completion * job_weight

        return current_value

    def export_data(self) -> str:
        scaled_weights = scale_to_int(self.job_weights) if self.scale_integer else self.job_weights
        types = "int" if self.scale_integer else "float"

        data = f"""
            job_weights = {scaled_weights};
        """

        return dedent(data)

    # Do not work
    def export_model(self) -> str:
        types = "int" if self.scale_integer else "float"

        model = f"""
            array[1..num_jobs] of {types}: job_weights;

            var {types}: weighted_completion_time;
            constraint weighted_completion_time = sum(j in 1..num_jobs)(job_weights[j] * max(t in 1..num_tasks where job[t] = j)(end[t, num_parts]));

            solve {self.direction} weighted_completion_time;
        """

        return dedent(model)

    def get_entry(self) -> str:
        return "Σ w_j C_j"

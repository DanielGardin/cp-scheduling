"""
objectives.py

This module defines the objective functions used in the scheduling environment.
Objective functions are used to evaluate the performance of a scheduling algorithm
and can be used to guide the search for an optimal schedule by providing a numerical value
that represents the quality of the schedule.
"""

from typing import Any
from collections.abc import Iterable
from typing_extensions import Self

from mypy_extensions import mypyc_attr

from ._common import TIME, Int, Float
from .data import SchedulingData
from .tasks import Tasks
from .utils import convert_to_list

objectives: dict[str, type["Objective"]] = {}


@mypyc_attr(allow_interpreted_subclasses=True)
class Objective:
    """
    Base class for all objective functions in the scheduling environment.

    Objective functions are used to evaluate the performance of a scheduling algorithm.
    They can be used to guide the search for an optimal schedule by providing a numerical value
    that represents the quality of the schedule.
    """

    minimize: bool

    tags: dict[str, str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        objectives[cls.__name__] = cls

    def __init__(self, minimize: bool = True) -> None:
        self.minimize = minimize
        self.tags = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def import_data(self, data: SchedulingData) -> None:
        "Import data from the instance when necessary."

    def export_data(self, data: SchedulingData) -> None:
        "Export data to the instance when necessary."

    def get_current(self, time: TIME, tasks: Tasks) -> float:
        """
        Get the current value of the objective function. This is useful for checking
        the performance of the scheduling algorithm along the episode.
        """
        return 0

    def __call__(
        self, time: int, tasks: Tasks, data: SchedulingData, objective: float
    ) -> float:
        "Call the objective function to get the current value."
        return self.get_current(time, tasks)

    def get_entry(self) -> str:
        "Produce the γ entry for the constraint."
        return ""

    def to_dict(self) -> dict[str, Any]:
        "Serialize the objective to a dictionary."
        return {
            "minimize": self.minimize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        "Deserialize the objective from a dictionary."
        return cls(**data)

    def __reduce__(self) -> tuple[Any, ...]:
        "Support for pickling the objective."
        return (self.__class__, tuple(self.to_dict().values()))


class ComposedObjective(Objective):
    """
    A composed objective function that combines multiple objectives with coefficients.

    This objective function allows for the combination of multiple objectives into a
    single objective function. Each objective can have a coefficient that scales its
    contribution to the overall objective value. The overall objective value is computed
    as a weighted sum of the individual objectives.

    Arguments:
        objectives: Iterable[Objective]
            An iterable of `Objective` instances to be combined.

        coefficients: Iterable[float], optional
            An iterable of coefficients for each objective. If not provided, all objectives
            are assumed to have a coefficient of 1.0.

        minimize: bool, default=True
            Whether to minimize or maximize the objective function.
    """

    objectives: list[Objective]
    coefficients: list[float]

    def __init__(
        self,
        objectives: Iterable[Objective],
        coefficients: Iterable[Float] | None = None,
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.objectives = list(objectives)

        self.coefficients = (
            [1.0] * len(self.objectives)
            if coefficients is None
            else convert_to_list(coefficients, float)
        )

    def import_data(self, data: SchedulingData) -> None:
        for objective in self.objectives:
            objective.import_data(data)

    def export_data(self, data: SchedulingData) -> None:
        for objective in self.objectives:
            objective.export_data(data)

    def get_current(self, time: TIME, tasks: Tasks) -> float:
        current_value = 0.0

        for objective, coefficient in zip(self.objectives, self.coefficients):
            current_value += coefficient * objective.get_current(time, tasks)

        return current_value

    def get_entry(self) -> str:
        entry = ""

        for coef, objective in zip(self.coefficients, self.objectives):
            if entry:
                if coef >= 0:
                    entry += " + "

                else:
                    entry += " - "
                    coef = -coef

            coef_str = (
                ""
                if coef == 1
                else str(coef) if isinstance(coef, int) else f"{coef:.2f}"
            )

            entry += f"{coef_str} {objective.get_entry()}"

        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": [
                objective.to_dict() | {"type": objective.__class__.__name__}
                for objective in self.objectives
            ],
            "coefficients": self.coefficients,
            "minimize": self.minimize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        global objectives

        obj_dict: list[dict[str, Any]] = data["objectives"]

        objectives_list = [
            objectives[objective_data.pop("type")].from_dict(objective_data)
            for objective_data in obj_dict
        ]

        return cls(
            objectives=objectives_list,
            coefficients=data["coefficients"],
            minimize=data["minimize"],
        )


class Makespan(Objective):
    """
    Classic makespan objective function, which aims to minimize the time at which all tasks are completed.
    """

    def get_current(self, time: TIME, tasks: Tasks) -> int:
        makespan = 0
        for task_id in tasks.fixed_tasks:
            task = tasks[task_id]

            if not task.is_completed(time):
                continue

            end_time = task.get_end()
            if end_time > makespan:
                makespan = end_time

        return makespan

    def get_entry(self) -> str:
        return "C_max"


class TotalCompletionTime(Objective):
    """
    The total completion time objective function, which aims to minimize the sum of completion times
    of all tasks.
    """

    def get_current(self, time: TIME, tasks: Tasks) -> int:
        total_completion_time = 0
        for job in range(tasks.n_jobs):
            job_completion = tasks.get_job_completion(job, time)
            total_completion_time += job_completion

        return total_completion_time

    def get_entry(self) -> str:
        return "ΣC_j"


class WeightedCompletionTime(Objective):
    """
    The weighted completion time objective function, which aims to minimize the weighted sum of
    completion times of all tasks. Each task has a weight associated with it, and the objective
    function is the sum of the completion times multiplied by their respective weights.
    """

    job_weights: list[float]

    def __init__(
        self,
        job_weights: Iterable[Float] | str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(job_weights, str):
            self.tags["job_weights"] = job_weights

        else:
            self.job_weights = convert_to_list(job_weights, float)

    def import_data(self, data: SchedulingData) -> None:
        if "job_weights" in self.tags:
            self.job_weights = data.get_job_level_data(self.tags["job_weights"])

    def export_data(self, data: SchedulingData) -> None:
        if not self.tags:
            data.add_data("weight", self.job_weights)

        else:
            data.add_alias("weight", self.tags["job_weights"])

    def get_current(self, time: TIME, tasks: Tasks) -> float:
        weighted_completion_time = 0.0
        for job in range(tasks.n_jobs):
            weight = self.job_weights[job]
            job_completion = tasks.get_job_completion(job, time)

            weighted_completion_time += weight * float(job_completion)

        return weighted_completion_time

    def get_entry(self) -> str:
        return "Σw_jC_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_weights": self.tags.get("job_weights", "weight"),
            "minimize": self.minimize,
        }


class MaximumLateness(Objective):
    """
    The maximum lateness objective function, which aims to minimize the maximum lateness of all
    tasks. Lateness is defined as the difference between the completion time and the due date.
    """

    due_dates: list[TIME]

    def __init__(
        self,
        due_dates: Iterable[Int] | str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(due_dates, str):
            self.tags["due_dates"] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, TIME)

    def import_data(self, data: SchedulingData) -> None:
        if "due_dates" in self.tags:
            self.due_dates = data.get_job_level_data(self.tags["due_dates"])

    def export_data(self, data: SchedulingData) -> None:
        if not self.tags:
            data.add_data("due_date", self.due_dates)

        else:
            data.add_alias("due_date", self.tags["due_dates"])

    def get_current(self, time: TIME, tasks: Tasks) -> int:
        max_lateness = 0

        for job in range(tasks.n_jobs):
            job_completion = tasks.get_job_completion(job, time)
            job_lateness = job_completion - self.due_dates[job]

            if max_lateness < job_lateness:
                max_lateness = job_lateness

        return max_lateness

    def get_entry(self) -> str:
        return "L_max"

    def to_dict(self) -> dict[str, Any]:
        return {
            "due_dates": self.tags.get("due_dates", "due_date"),
            "minimize": self.minimize,
        }


class TotalTardiness(Objective):
    """
    The total tardiness objective function, which aims to minimize the sum of tardiness of all
    tasks. Tardiness is defined as the difference between the completion time and the due date,
    if the task is completed late.
    """

    due_dates: list[TIME]

    def __init__(
        self,
        due_dates: Iterable[Int] | str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(due_dates, str):
            self.tags["due_dates"] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, TIME)

    def import_data(self, data: SchedulingData) -> None:
        if "due_dates" in self.tags:
            self.due_dates = data.get_job_level_data(self.tags["due_dates"])

    def export_data(self, data: SchedulingData) -> None:
        if not self.tags:
            data.add_data("due_date", self.due_dates)

        else:
            data.add_alias("due_date", self.tags["due_dates"])

    def get_current(self, time: TIME, tasks: Tasks) -> int:
        total_tardiness = 0
        for job in range(tasks.n_jobs):
            due_date = self.due_dates[job]
            job_completion = tasks.get_job_completion(job, time)

            job_tardiness = (
                job_completion - due_date if job_completion > due_date else 0
            )

            total_tardiness += job_tardiness

        return total_tardiness

    def get_entry(self) -> str:
        return "ΣT_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "due_dates": self.tags.get("due_dates", "due_date"),
            "minimize": self.minimize,
        }


class WeightedTardiness(Objective):
    """
    The weighted tardiness objective function, which aims to minimize the weighted sum of tardiness
    of all tasks. Tardiness is defined as the difference between the completion time and the due
    date, if the task is completed late.
    """

    due_dates: list[TIME]
    job_weights: list[float]

    def __init__(
        self,
        due_dates: Iterable[Int] | str = "due_date",
        job_weights: Iterable[Float] | str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(due_dates, str):
            self.tags["due_dates"] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, TIME)

        if isinstance(job_weights, str):
            self.tags["job_weights"] = job_weights

        else:
            self.job_weights = convert_to_list(job_weights, float)

    def import_data(self, data: SchedulingData) -> None:
        if "due_dates" in self.tags:
            self.due_dates = data.get_job_level_data(self.tags["due_dates"])

        if "job_weights" in self.tags:
            self.job_weights = data.get_job_level_data(self.tags["job_weights"])

    def export_data(self, data: SchedulingData) -> None:
        if "due_dates" not in self.tags:
            data.add_data("due_date", self.due_dates)

        else:
            data.add_alias("due_date", self.tags["due_dates"])

        if "job_weights" not in self.tags:
            data.add_data("weight", self.job_weights)

        else:
            data.add_alias("weight", self.tags["job_weights"])

    def get_current(self, time: TIME, tasks: Tasks) -> float:
        weighted_tardiness = 0.0
        for job in range(tasks.n_jobs):
            weight = self.job_weights[job]
            due_date = self.due_dates[job]
            job_completion = tasks.get_job_completion(job, time)

            job_tardiness = (
                float(job_completion - due_date) if job_completion > due_date else 0.0
            )

            weighted_tardiness += weight * job_tardiness

        return weighted_tardiness

    def get_entry(self) -> str:
        return "Σw_jT_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "due_dates": self.tags.get("due_dates", "due_date"),
            "job_weights": self.tags.get("job_weights", "weight"),
            "minimize": self.minimize,
        }


class TotalEarliness(Objective):
    """
    The total earliness objective function, which aims to minimize the sum of earliness of all
    tasks. Earliness is defined as the difference between the due date and the completion time,
    if the task is completed early.
    """

    due_dates: list[TIME]

    def __init__(
        self,
        due_dates: Iterable[Int] | str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(due_dates, str):
            self.tags["due_dates"] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, TIME)

    def import_data(self, data: SchedulingData) -> None:
        if "due_dates" in self.tags:
            self.due_dates = data.get_job_level_data(self.tags["due_dates"])

    def export_data(self, data: SchedulingData) -> None:
        if not self.tags:
            data.add_data("due_date", self.due_dates)

        else:
            data.add_alias("due_date", self.tags["due_dates"])

    def get_current(self, time: TIME, tasks: Tasks) -> int:
        total_earliness = 0
        for job in range(tasks.n_jobs):
            due_date = self.due_dates[job]
            job_completion = tasks.get_job_completion(job, time)

            job_earliness = (
                due_date - job_completion if job_completion < due_date else 0
            )

            total_earliness += job_earliness

        return total_earliness

    def get_entry(self) -> str:
        return "ΣE_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "due_dates": self.tags.get("due_dates", "due_date"),
            "minimize": self.minimize,
        }


class WeightedEarliness(Objective):
    """
    The weighted earliness objective function, which aims to minimize the weighted sum of earliness of all tasks.
    Earliness is defined as the difference between the due date and the completion time, if the task is completed early.
    """

    due_dates: list[TIME]
    job_weights: list[float]

    def __init__(
        self,
        due_dates: Iterable[Int] | str = "due_date",
        job_weights: Iterable[Float] | str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(due_dates, str):
            self.tags["due_dates"] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, TIME)

        if isinstance(job_weights, str):
            self.tags["job_weights"] = job_weights

        else:
            self.job_weights = convert_to_list(job_weights, float)

    def import_data(self, data: SchedulingData) -> None:
        if "due_dates" in self.tags:
            self.due_dates = data.get_job_level_data(self.tags["due_dates"])

        if "job_weights" in self.tags:
            self.job_weights = data.get_job_level_data(self.tags["job_weights"])

    def export_data(self, data: SchedulingData) -> None:
        if "due_dates" not in self.tags:
            data.add_data("due_date", self.due_dates)

        else:
            data.add_alias("due_date", self.tags["due_dates"])

        if "job_weights" not in self.tags:
            data.add_data("weight", self.job_weights)

        else:
            data.add_alias("weight", self.tags["job_weights"])

    def get_current(self, time: TIME, tasks: Tasks) -> float:
        weighted_earliness = 0.0
        for job in range(tasks.n_jobs):
            weight = self.job_weights[job]
            due_date = self.due_dates[job]
            job_completion = tasks.get_job_completion(job, time)

            job_earliness = (
                float(due_date - job_completion) if job_completion < due_date else 0
            )

            weighted_earliness += weight * job_earliness

        return weighted_earliness

    def get_entry(self) -> str:
        return "Σw_jE_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "due_dates": self.tags.get("due_dates", "due_date"),
            "job_weights": self.tags.get("job_weights", "weight"),
            "minimize": self.minimize,
        }


class TotalTardyJobs(Objective):
    """
    The total tardy jobs objective function, which aims to minimize the number of tardy jobs.
    A job is considered tardy if its completion time is greater than its due date.
    """

    due_dates: list[TIME]

    def __init__(
        self,
        due_dates: Iterable[Int] | str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(due_dates, str):
            self.tags["due_dates"] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, TIME)

    def import_data(self, data: SchedulingData) -> None:
        if "due_dates" in self.tags:
            self.due_dates = convert_to_list(
                data.get_job_level_data(self.tags["due_dates"]), TIME
            )

    def export_data(self, data: SchedulingData) -> None:
        if not self.tags:
            data.add_data("due_date", self.due_dates)

        else:
            data.add_alias("due_date", self.tags["due_dates"])

    def get_current(self, time: TIME, tasks: Tasks) -> int:
        tardy_jobs = 0
        for job in range(tasks.n_jobs):
            due_date = self.due_dates[job]
            job_completion = tasks.get_job_completion(job, time)

            tardy = 1 if job_completion > due_date else 0
            tardy_jobs += tardy

        return tardy_jobs

    def get_entry(self) -> str:
        return "ΣU_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "due_dates": self.tags.get("due_dates", "due_date"),
            "minimize": self.minimize,
        }


class WeightedTardyJobs(Objective):
    """
    The weighted tardy jobs objective function, which aims to minimize the weighted sum of tardy
    jobs. A job is considered tardy if its completion time is greater than its due date.
    """

    due_dates: list[TIME]
    job_weights: list[float]

    def __init__(
        self,
        due_dates: Iterable[Int] | str = "due_date",
        job_weights: Iterable[Float] | str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(due_dates, str):
            self.tags["due_dates"] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, TIME)

        if isinstance(job_weights, str):
            self.tags["job_weights"] = job_weights

        else:
            self.job_weights = convert_to_list(job_weights, float)

    def import_data(self, data: SchedulingData) -> None:
        if "due_dates" in self.tags:
            self.due_dates = data.get_job_level_data(self.tags["due_dates"])

        if "job_weights" in self.tags:
            self.job_weights = data.get_job_level_data(self.tags["job_weights"])

    def export_data(self, data: SchedulingData) -> None:
        if "due_dates" not in self.tags:
            data.add_data("due_date", self.due_dates)

        else:
            data.add_alias("due_date", self.tags["due_dates"])

        if "job_weights" not in self.tags:
            data.add_data("weight", self.job_weights)

        else:
            data.add_alias("weight", self.tags["job_weights"])

    def get_current(self, time: TIME, tasks: Tasks) -> float:
        weighted_tardy_jobs = 0.0
        for job in range(tasks.n_jobs):
            weight = self.job_weights[job]
            due_date = self.due_dates[job]
            job_completion = tasks.get_job_completion(job, time)

            tardy = weight if job_completion > due_date else 0.0
            weighted_tardy_jobs += tardy

        return weighted_tardy_jobs

    def get_entry(self) -> str:
        return "Σw_jU_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "due_dates": self.tags.get("due_dates", "due_date"),
            "job_weights": self.tags.get("job_weights", "weight"),
            "minimize": self.minimize,
        }


class TotalFlowTime(Objective):
    """
    The total flow time objective function, which aims to minimize the sum of flow times of all
    tasks. Flow time is defined as the difference between the completion time and the release time.
    """

    release_times: list[TIME]

    def __init__(
        self,
        release_times: Iterable[Int] | str = "release_time",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        if isinstance(release_times, str):
            self.tags["release_times"] = release_times

        else:
            self.release_times = convert_to_list(release_times, TIME)

    def import_data(self, data: SchedulingData) -> None:
        if "release_times" in self.tags:
            self.release_times = data.get_job_level_data(self.tags["release_times"])

    def export_data(self, data: SchedulingData) -> None:
        if not self.tags:
            data.add_data("release_time", self.release_times)

        else:
            data.add_alias("release_time", self.tags["release_times"])

    def get_current(self, time: TIME, tasks: Tasks) -> int:
        total_flowtime = 0
        for job in range(tasks.n_jobs):
            release_time = self.release_times[job]
            job_completion = tasks.get_job_completion(job, time)

            job_flowtime = (
                job_completion - release_time if job_completion > release_time else 0
            )

            total_flowtime += job_flowtime

        return total_flowtime

    def get_entry(self) -> str:
        return "ΣF_j"

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_times": self.tags.get("release_times", "release_time"),
            "minimize": self.minimize,
        }

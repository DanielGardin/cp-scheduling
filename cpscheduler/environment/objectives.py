"""
    objectives.py

    This module defines the objective functions used in the scheduling environment.
    Objective functions are used to evaluate the performance of a scheduling algorithm
    and can be used to guide the search for an optimal schedule by providing a numerical value
    that represents the quality of the schedule.
"""
from typing import ClassVar, Any
from collections.abc import Iterable

from mypy_extensions import mypyc_attr

from .tasks import Tasks
from .utils import convert_to_list

objectives: dict[str, type['Objective']] = {}

@mypyc_attr(allow_interpreted_subclasses=True)
class Objective:
    """
    Base class for all objective functions in the scheduling environment.

    Objective functions are used to evaluate the performance of a scheduling algorithm.
    They can be used to guide the search for an optimal schedule by providing a numerical value
    that represents the quality of the schedule.
    """
    default_minimize: bool = True
    tags: dict[str, str]  = {}

    tasks: Tasks
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        objectives[cls.__name__] = cls

    def __init__(self) -> None:
        self.loaded = False
        self.tags = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self.loaded})"

    def set_tasks(self, tasks: Tasks) -> None:
        "Make the objective aware of the tasks it is applied to."
        self.tasks = tasks
        self.loaded = True

    def has_tag(self, tag: str) -> bool:
        "Check if the constraint have a tag defined to search on the tasks data."
        has_tag = tag in self.tags

        if has_tag and self.loaded and self.tags[tag] not in self.tasks.data:
            raise ValueError(f"Tag '{tag}' not found in tasks data.")

        return has_tag

    def get_data(self, feature_or_tag: str) -> list[Any]:
        "Get the data for a feature or tag from the tasks data."
        feature = self.tags.get(feature_or_tag, feature_or_tag)
        return self.tasks.get_data(feature)

    def get_current(self, time: int) -> float:
        """
        Get the current value of the objective function. This is useful for checking
        the performance of the scheduling algorithm along the episode.
        """
        return 0

    def get_entry(self) -> str:
        "Produce the γ entry for the constraint."
        return ""


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
    """
    def __init__(
        self,
        objectives: Iterable[Objective],
        coefficients: Iterable[float] | None = None
    ):
        super().__init__()
        self.objectives = list(objectives)

        self.coefficients = (
            [1.0] * len(self.objectives) if coefficients is None
            else convert_to_list(coefficients)
        )

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        for objective in self.objectives:
            objective.set_tasks(tasks)

    def get_current(self, time: int) -> float:
        current_value = 0.0

        for objective, coefficient in zip(self.objectives, self.coefficients):
            current_value += coefficient * objective.get_current(time)

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
                ""               if coef == 1 else
                str(coef) if isinstance(coef, int) else
                f"{coef:.2f}"
            )

            entry += f"{coef_str} {objective.get_entry()}"

        return entry

class Makespan(Objective):
    """
    Classic makespan objective function, which aims to minimize the time at which all tasks are completed.
    """
    objective_name: ClassVar[str] = "makespan"

    def get_current(self, time: int) -> int:
        task_ends = [task.get_end() for task in self.tasks if task.is_completed(time)]

        return max(task_ends, default=0)

    def get_entry(self) -> str:
        return "C_max"

class TotalCompletionTime(Objective):
    """
    The total completion time objective function, which aims to minimize the sum of completion times
    of all tasks.
    """
    def get_current(self, time: int) -> int:
        completion_times = [
            max([task.get_end() for task in tasks if task.is_completed(time)], default=0)
            for tasks in self.tasks.jobs
        ]

        return sum(completion_times)

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
        job_weights: Iterable[float] | str = 'weight',
    ):
        super().__init__()
        if isinstance(job_weights, str):
            self.tags['job_weights'] = job_weights

        else:
            self.job_weights = convert_to_list(job_weights)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'job_weights' in self.tags:
            self.job_weights = self.get_data('job_weights')

    def get_current(self, time: int) -> float:
        weighted_completion_time = 0.

        for job, tasks in enumerate(self.tasks.jobs):
            weight   = self.job_weights[job]

            max_completion_time = max(
                [task.get_end() for task in tasks if task.is_completed(time)],
                default=0
            )

            weighted_completion_time += weight * max_completion_time

        return weighted_completion_time

    def get_entry(self) -> str:
        return "Σw_jC_j"

class MaximumLateness(Objective):
    """
    The maximum lateness objective function, which aims to minimize the maximum lateness of all
    tasks. Lateness is defined as the difference between the completion time and the due date.
    """
    due_dates: list[int]

    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
    ):
        super().__init__()
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

    def get_current(self, time: int) -> float:
        lateness = [
            task.get_end() - due_date
            for due_date, task in zip(self.due_dates, self.tasks)
            if task.is_completed(time)
        ]

        return max(lateness, default=0)

    def get_entry(self) -> str:
        return "L_max"


class TotalTardiness(Objective):
    """
    The total tardiness objective function, which aims to minimize the sum of tardiness of all
    tasks. Tardiness is defined as the difference between the completion time and the due date,
    if the task is completed late.
    """
    due_dates: list[int]

    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
    ):
        super().__init__()
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

    def get_current(self, time: int) -> float:
        tardiness = [
            max(
                0,
                max([task.get_end() for task in tasks if task.is_completed(time)], default=0) -
                due_date
            )
            for due_date, tasks in zip(self.due_dates, self.tasks.jobs)
        ]

        return sum(tardiness)

    def get_entry(self) -> str:
        return "ΣT_j"

class WeightedTardiness(Objective):
    """
    The weighted tardiness objective function, which aims to minimize the weighted sum of tardiness
    of all tasks. Tardiness is defined as the difference between the completion time and the due
    date, if the task is completed late.
    """
    job_weights: list[float]

    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
        job_weights: Iterable[float] | str = 'weight',
    ):
        super().__init__()
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, int)

        if isinstance(job_weights, str):
            self.tags['job_weights'] = job_weights

        else:
            self.job_weights = convert_to_list(job_weights, float)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

        if 'job_weights' in self.tags:
            self.job_weights = self.get_data('job_weights')

    def get_current(self, time: int) -> float:
        weighted_tardiness = 0.

        for job, tasks in enumerate(self.tasks.jobs):
            weight   = self.job_weights[job]
            due_date = self.due_dates[job]

            max_completion_time = max(
                [task.get_end() for task in tasks if task.is_completed(time)],
                default=0
            )

            weighted_tardiness += weight * max(0, max_completion_time - due_date)

        return weighted_tardiness

    def get_entry(self) -> str:
        return "Σw_jT_j"

class TotalEarliness(Objective):
    """
    The total earliness objective function, which aims to minimize the sum of earliness of all
    tasks. Earliness is defined as the difference between the due date and the completion time,
    if the task is completed early.
    """
    due_dates: list[int]

    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
    ):
        super().__init__()
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

    def get_current(self, time: int) -> float:
        tardiness = [
            max(
                0,
                due_date -
                max([task.get_end() for task in tasks if task.is_completed(time)], default=0)
            ) for due_date, tasks in zip(self.due_dates, self.tasks.jobs)
        ]

        return sum(tardiness)

    def get_entry(self) -> str:
        return "ΣE_j"

class WeightedEarliness(Objective):
    """
    The weighted earliness objective function, which aims to minimize the weighted sum of earliness of all tasks.
    Earliness is defined as the difference between the due date and the completion time, if the task is completed early.
    """
    due_dates: list[int]
    job_weights: list[float]

    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
        job_weights: Iterable[float] | str = 'weight',
    ):
        super().__init__()
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, int)

        if isinstance(job_weights, str):
            self.tags['job_weights'] = job_weights

        else:
            self.job_weights = convert_to_list(job_weights, float)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

        if 'job_weights' in self.tags:
            self.job_weights = self.get_data('job_weights')

    def get_current(self, time: int) -> float:
        weighted_tardiness = 0.

        for job, tasks in enumerate(self.tasks.jobs):
            weight   = self.job_weights[job]
            due_date = self.due_dates[job]

            max_completion_time = max(
                [task.get_end() for task in tasks if task.is_completed(time)],
                default=0
            )

            weighted_tardiness += weight * max(0, due_date - max_completion_time)

        return weighted_tardiness

    def get_entry(self) -> str:
        return "Σw_jE_j"

class TotalTardyJobs(Objective):
    """
    The total tardy jobs objective function, which aims to minimize the number of tardy jobs.
    A job is considered tardy if its completion time is greater than its due date.
    """
    due_dates: list[int]

    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
    ):
        super().__init__()
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

    def get_current(self, time: int) -> float:
        tardiness = [
            1 if max(
                [task.get_end() for task in tasks if task.is_completed(time)],
                default=0
            ) > due_date else 0
            for due_date, tasks in zip(self.due_dates, self.tasks.jobs)
        ]

        return sum(tardiness)

    def get_entry(self) -> str:
        return "ΣU_j"

class WeightedTardyJobs(Objective):
    """
    The weighted tardy jobs objective function, which aims to minimize the weighted sum of tardy
    jobs. A job is considered tardy if its completion time is greater than its due date.
    """
    due_dates: list[int]
    job_weights: list[float]

    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
        job_weights: Iterable[float] | str = 'weight',
    ):
        super().__init__()
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates

        else:
            self.due_dates = convert_to_list(due_dates, int)

        if isinstance(job_weights, str):
            self.tags['job_weights'] = job_weights

        else:
            self.job_weights = convert_to_list(job_weights, float)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

        if 'job_weights' in self.tags:
            self.job_weights = self.get_data('job_weights')

    def get_current(self, time: int) -> float:
        weighted_tardiness = 0.

        for job, tasks in enumerate(self.tasks.jobs):
            weight   = self.job_weights[job]
            due_date = self.due_dates[job]

            max_completion_time = max(
                [task.get_end() for task in tasks if task.is_completed(time)],
                default=0
            )

            weighted_tardiness += weight * (1 if max_completion_time > due_date else 0)

        return weighted_tardiness

    def get_entry(self) -> str:
        return "Σw_jU_j"

class TotalFlowTime(Objective):
    """
    The total flow time objective function, which aims to minimize the sum of flow times of all
    tasks. Flow time is defined as the difference between the completion time and the release time.
    """
    release_times: list[int]

    def __init__(
        self,
        release_times: Iterable[int] | str = 'release_time',
    ):
        super().__init__()
        if isinstance(release_times, str):
            self.tags['release_times'] = release_times

        else:
            self.release_times = convert_to_list(release_times, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'release_times' in self.tags:
            self.release_times = self.get_data('release_times')

    def get_current(self, time: int) -> float:
        flow_time = [
            max(
                0,
                max([task.get_end() for task in tasks if task.is_completed(time)], default=0) -
                release_time
            )
            for release_time, tasks in zip(self.release_times, self.tasks.jobs)
        ]

        return sum(flow_time)

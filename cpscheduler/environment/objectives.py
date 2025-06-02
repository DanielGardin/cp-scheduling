from typing import ClassVar, Iterable, Optional, Any

from textwrap import dedent

from .tasks import Tasks
from .utils import convert_to_list, scale_to_int

class Objective:
    default_minimize: bool = True
    objective_name: ClassVar[str] = "objective"
    tags: dict[str, str]  = {}

    tasks: Tasks

    def set_tasks(self, tasks: Tasks) -> None:
        self.tasks = tasks

    def get_current(self, time: int) -> float:
        """
        Get the current value of the objective function. This is useful for checking the performance of the
        scheduling algorithm along the episode.
        """
        return 0

    def get_data(self, feature: str) -> list[Any]:
        feature = self.tags.get(feature, feature)

        if feature in self.tasks.data:
            return self.tasks.data[feature]

        if feature in self.tasks.jobs_data:
            data = self.tasks.jobs_data[feature]

            return [data[job] for job in self.tasks.data['job_id']]

        assert False, f"Feature {feature} not found in tasks data"

    def get_entry(self) -> str:
        return ""


class ComposedObjective(Objective):
    """
        Compose a set of objective functions into a single objective function by a linear combination of their values.
    """
    objective_name: ClassVar[str] = "composed_objective"

    def __init__(
        self,
        objectives: Iterable[Objective],
        coefficients: Optional[Iterable[float]] = None
    ):
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
        return " + ".join([f"a_{i} * {objective.get_entry()}" for i, objective in enumerate(self.objectives)])

class Makespan(Objective):
    """
    Classic makespan objective function, which aims to minimize the time at which all tasks are completed.
    """
    objective_name: ClassVar[str] = "makespan"

    def get_current(self, time: int) -> int:
        task_ends = [task.get_end() for task in self.tasks if task.is_completed(time)]

        return max(task_ends, default=0)

    def export_model(self) -> str:
        model = f"""
            var int: makespan;
            constraint makespan = max(t in 1..num_tasks)(end[t, num_parts]);
        """

        return dedent(model)

    def get_entry(self) -> str:
        return "C_max"


class TotalCompletionTime(Objective):
    """
    The total completion time objective function, which aims to minimize the sum of completion times of all tasks.
    """
    objective_name: ClassVar[str] = "total_completion_time"

    def get_current(self, time: int) -> int:
        completion_times = [
            max([task.get_end() for task in tasks if task.is_completed(time)], default=0)
            for tasks in self.tasks.jobs
        ]

        return sum(completion_times)

    def export_model(self) -> str:
        model = f"""
            var int: total_completion_time;
            constraint total_completion_time = sum(t in 1..num_tasks)(end[t, num_parts]);
        """

        return dedent(model)

    def get_entry(self) -> str:
        return "Σ C_j"

class WeightedCompletionTime(Objective):
    """
    The weighted completion time objective function, which aims to minimize the weighted sum of completion times of all tasks.
    Each task has a weight associated with it, and the objective function is the sum of the completion times multiplied by their respective weights.
    """
    objective_name: ClassVar[str] = "weighted_completion_time"

    def __init__(
        self,
        job_weights: Iterable[float] | str = 'weight',
    ):
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

        for job in range(len(self.tasks.jobs)):
            weight   = self.job_weights[job]
            tasks    = self.tasks.jobs[job]

            max_completion_time = max([task.get_end() for task in tasks if task.is_completed(time)], default=0)

            weighted_completion_time += weight * max_completion_time

        return weighted_completion_time

    # def export_data(self, scale_int: bool = False) -> str:
    #     scaled_weights = scale_to_int(self.job_weights) if scale_int else self.job_weights

    #     data = f"""
    #         job_weights = {scaled_weights};
    #     """

    #     return dedent(data)

    # # Do not work
    # def export_model(self) -> str:
    #     types = "int" if self.scale_integer else "float"

    #     model = f"""
    #         array[1..num_jobs] of {types}: job_weights;

    #         var {types}: weighted_completion_time;
    #         constraint weighted_completion_time = sum(j in 1..num_jobs)(job_weights[j] * max(t in job[j])(end[t, num_parts]));
    #     """

    #     return dedent(model)

    def get_entry(self) -> str:
        return "Σ w_j C_j"

class MaximumLateness(Objective):
    """
    The maximum lateness objective function, which aims to minimize the maximum lateness of all tasks.
    Lateness is defined as the difference between the completion time and the due date.
    """
    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
    ):
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates
            self.due_dates = []
        
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

    # def export_model(self) -> str:
    #     model = f"""
    #         var int: maximum_lateness;
    #         constraint maximum_lateness = max(t in 1..num_tasks)(end[t, num_parts] - due_dates[t]);

    #         {self.solve_annotation} {self.direction} maximum_lateness;
    #     """

    #     return dedent(model)
    
    # def export_data(self) -> str:
    #     data = f"""
    #         due_dates = {self.due_dates};
    #     """

    #     return dedent(data)
    
    def get_entry(self) -> str:
        return "L_max"


class TotalTardiness(Objective):
    """
    The total tardiness objective function, which aims to minimize the sum of tardiness of all tasks.
    Tardiness is defined as the difference between the completion time and the due date, if the task is completed late.
    """
    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
    ):
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates
            self.due_dates = []
        
        else:
            self.due_dates = convert_to_list(due_dates, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

    def get_current(self, time: int) -> float:
        tardiness = [
            max(0, max([task.get_end() for task in tasks if task.is_completed(time)], default=0) - due_date)
            for due_date, tasks in zip(self.due_dates, self.tasks.jobs)
        ]

        return sum(tardiness)


class WeightedTardiness(Objective):
    """
    The weighted tardiness objective function, which aims to minimize the weighted sum of tardiness of all tasks.
    Tardiness is defined as the difference between the completion time and the due date, if the task is completed late.
    """
    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
        job_weights: Iterable[float] | str = 'weight',
    ):
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates
            self.due_dates = []
        
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

        for job in range(len(self.tasks.jobs)):
            weight   = self.job_weights[job]
            tasks    = self.tasks.jobs[job]
            due_date = self.due_dates[job]

            max_completion_time = max([task.get_end() for task in tasks if task.is_completed(time)], default=0)

            weighted_tardiness += weight * max(0, max_completion_time - due_date)

        return weighted_tardiness
    
    # def export_model(self) -> str:
    #     types = "int" if self.scale_integer else "float"

    #     model = f"""
    #         array[1..num_tasks] of {types}: due_dates;
    #         array[1..num_tasks] of {types}: weights;

    #         var {types}: weighted_tardiness;
    #         constraint weighted_tardiness = sum(t in 1..num_tasks)(max(0, end[t, num_parts] - due_dates[t]) * weights[t]);

    #         {self.solve_annotation} {self.direction} weighted_tardiness;
    #     """

    #     return dedent(model)
    
    # def export_data(self) -> str:
    #     scaled_weights = scale_to_int(self.weights) if self.scale_integer else self.weights
    #     types = "int" if self.scale_integer else "float"

    #     data = f"""
    #         due_dates = {self.due_dates};
    #         weights = {scaled_weights};
    #     """

    #     return dedent(data)
    
    def get_entry(self) -> str:
        return "Σ w_j T_j"


class TotalEarliness(Objective):
    """
    The total earliness objective function, which aims to minimize the sum of earliness of all tasks.
    Earliness is defined as the difference between the due date and the completion time, if the task is completed early.
    """
    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
    ):
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates
            self.due_dates = []
        
        else:
            self.due_dates = convert_to_list(due_dates, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

    def get_current(self, time: int) -> float:
        tardiness = [
            max(0, due_date - max([task.get_end() for task in tasks if task.is_completed(time)], default=0))
            for due_date, tasks in zip(self.due_dates, self.tasks.jobs)
        ]

        return sum(tardiness)


class WeightedEarliness(Objective):
    """
    The weighted earliness objective function, which aims to minimize the weighted sum of earliness of all tasks.
    Earliness is defined as the difference between the due date and the completion time, if the task is completed early.
    """
    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
        job_weights: Iterable[float] | str = 'weight',
    ):
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates
            self.due_dates = []
        
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

        for job in range(len(self.tasks.jobs)):
            weight   = self.job_weights[job]
            tasks    = self.tasks.jobs[job]
            due_date = self.due_dates[job]

            max_completion_time = max([task.get_end() for task in tasks if task.is_completed(time)], default=0)

            weighted_tardiness += weight * max(0, due_date - max_completion_time)

        return weighted_tardiness


class TotalTardyJobs(Objective):
    """
    The total tardy jobs objective function, which aims to minimize the number of tardy jobs.
    A job is considered tardy if its completion time is greater than its due date.
    """
    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
    ):
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates
            self.due_dates = []
        
        else:
            self.due_dates = convert_to_list(due_dates, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'due_dates' in self.tags:
            self.due_dates = self.get_data('due_dates')

    def get_current(self, time: int) -> float:
        tardiness = [
            1 if max([task.get_end() for task in tasks if task.is_completed(time)], default=0) > due_date else 0
            for due_date, tasks in zip(self.due_dates, self.tasks.jobs)
        ]

        return sum(tardiness)
    
class WeightedTardyJobs(Objective):
    """
    The weighted tardy jobs objective function, which aims to minimize the weighted sum of tardy jobs.
    A job is considered tardy if its completion time is greater than its due date.
    """
    def __init__(
        self,
        due_dates: Iterable[int] | str = 'due_date',
        job_weights: Iterable[float] | str = 'weight',
    ):
        if isinstance(due_dates, str):
            self.tags['due_dates'] = due_dates
            self.due_dates = []
        
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

        for job in range(len(self.tasks.jobs)):
            weight   = self.job_weights[job]
            tasks    = self.tasks.jobs[job]
            due_date = self.due_dates[job]

            max_completion_time = max([task.get_end() for task in tasks if task.is_completed(time)], default=0)

            weighted_tardiness += weight * (1 if max_completion_time > due_date else 0)

        return weighted_tardiness


class TotalFlowTime(Objective):
    """
    The total flow time objective function, which aims to minimize the sum of flow times of all tasks.
    Flow time is defined as the difference between the completion time and the release time.
    """
    def __init__(
        self,
        release_times: Iterable[int] | str = 'release_time',
    ):
        if isinstance(release_times, str):
            self.tags['release_times'] = release_times
            self.release_times = []
        
        else:
            self.release_times = convert_to_list(release_times, int)

    def set_tasks(self, tasks: Tasks) -> None:
        super().set_tasks(tasks)

        if 'release_times' in self.tags:
            self.release_times = self.get_data('release_times')

    def get_current(self, time: int) -> float:
        flow_time = [
            max(0, max([task.get_end() for task in tasks if task.is_completed(time)], default=0) - release_time)
            for release_time, tasks in zip(self.release_times, self.tasks.jobs)
        ]

        return sum(flow_time)
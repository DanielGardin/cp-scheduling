"""Makespan and maximum lateness objectives."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import Int, JobID, Time
from cpscheduler.environment.instance import Feature
from cpscheduler.environment.objectives.base import MaxCompletionObjective
from cpscheduler.environment.utils import convert_to_list


class Makespan(MaxCompletionObjective):
    """Makespan objective.

    This objective function aims to minimize the time at which all tasks are completed.
    """

    @override
    def job_cost(self, job_id: JobID, completion: Time) -> float:
        return float(completion)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "C_max"


class MaximumLateness(MaxCompletionObjective):
    """Maximum Lateness objective.

    This objective function aims to minimize the maximum lateness of all jobs.
    Lateness of a job is defined as the amount of time by which its completion time
    exceeds its due date, i.e., L_j = C_j - d_j
    """

    due_dates: Feature[list[Time]]

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
    ):
        """Initialize the Maximum Lateness objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: Iterable[Time] | None, optional
            The due dates for each job.
            If None, due dates must be provided in the instance data.

        """
        super().__init__()

        self.due_dates = Feature(
            name=due_dates_tag,
            preprocess=self._load_dates,
            shape=("n_jobs",),
            value_type="time",
        )

        if due_dates is not None:
            self.due_dates.own_data(due_dates)

    def _load_dates(self, dates: Iterable[Int]) -> list[Time]:
        return convert_to_list(dates, Time)

    @override
    def job_cost(self, job_id: JobID, completion: Time) -> float:
        return float(completion - self.due_dates.value[job_id])

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "L_max"

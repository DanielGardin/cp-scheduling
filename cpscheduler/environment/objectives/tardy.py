"""Tardy jobs objectives."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import (
    Float,
    Int,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.instance import Feature, ProblemInstance
from cpscheduler.environment.objectives.base import CompletionTimeObjective
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class TotalTardyJobs(CompletionTimeObjective):
    """Total Tardy Jobs objective.

    This objective function aims to minimize the number of tardy jobs.
    A job is tardy if its completion time exceeds its due date, i.e., C_j > d_j.
    """

    due_dates: Feature[list[Time]]

    _tardy_jobs: list[bool]
    _n_tardy_jobs: int

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
        minimize: bool = True,
    ) -> None:
        """Initialize the Total Tardy Jobs objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: Iterable[Time] | None, optional
            The due dates for each job.
            If None, due dates must be provided in the instance data.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

        self.due_dates = Feature(
            name=due_dates_tag,
            preprocess=self._load_due_dates,
            shape=("n_jobs",),
        )

    def _load_due_dates(self, due_dates: Iterable[Int]) -> list[Time]:
        return convert_to_list(due_dates, Time)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)

        self._tardy_jobs = [False] * instance.n_jobs
        self._n_tardy_jobs = 0

    @override
    def reset(self, state: ScheduleState) -> None:
        super().reset(state)

        for job_id in range(state.n_jobs):
            self._tardy_jobs[job_id] = False

        self._n_tardy_jobs = 0

    @override
    def on_task_completed(
        self,
        task_id: TaskID,
        machine_id: MachineID,
        state: ScheduleState,
    ) -> None:
        super().on_task_completed(task_id, machine_id, state)

        job_id = state.instance.job_ids[task_id]

        if self._tardy_jobs[job_id]:
            return

        C_j = self._job_completion[job_id]
        d_j = self.due_dates.value[job_id]

        if C_j > d_j:
            self._tardy_jobs[job_id] = True
            self._n_tardy_jobs += 1

    @override
    def get_current(self, state: ScheduleState) -> float:
        return float(self._n_tardy_jobs)

    @override
    def __call__(self, state: ScheduleState) -> float:
        return float(
            sum(
                C_j > d_j
                for C_j, d_j in zip(
                    self.completion_times(state),
                    self.due_dates.value,
                    strict=False,
                )
            )
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "ΣU_j"


class WeightedTardyJobs(TotalTardyJobs):
    """Weighted Tardy Jobs objective.

    This objective function aims to minimize the weighted number of tardy jobs.
    A job is tardy if its completion time exceeds its due date, i.e., C_j > d_j.
    The weighted variant optimizes Σw_jU_j, where w_j is the weight of job j.
    """

    weights: Feature[list[float]]

    _weighted_tardy_jobs: float

    def __init__(
        self,
        due_dates_tag: str = "due_date",
        due_dates: Iterable[Int] | None = None,
        weights_tag: str = "weight",
        weights: Iterable[Float] | None = None,
        minimize: bool = True,
    ) -> None:
        """Initialize the Weighted Tardy Jobs objective.

        Parameters
        ----------
        due_dates_tag: str, optional
            The name of the job feature that contains the due dates.

        due_dates: Iterable[Time] | None, optional
            The due dates for each job.
            If None, due dates must be provided in the instance data.

        weights_tag: str, optional
            The name of the job feature that contains the weights.
            Default to "weight".

        weights: Iterable[Float] | None, optional
            The weights for each job.
            If None is provided, the weights will be loaded from the instance.

        minimize: bool, optional
            Whether to minimize or maximize the objective.
            Default is True (i.e., minimize).

        """
        super().__init__(due_dates_tag, due_dates, minimize)

        self.weights = Feature(
            name=weights_tag,
            preprocess=self._load_weights,
            shape=("n_jobs",),
        )

        if weights is not None:
            self.weights.own_data(weights)

    def _load_weights(self, weights: Iterable[Float]) -> list[float]:
        return convert_to_list(weights, float)

    @property
    @override
    def regular(self) -> bool:
        return all(weight >= 0.0 for weight in self.weights.value)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)

        self._weighted_tardy_jobs = 0.0

    @override
    def reset(self, state: ScheduleState) -> None:
        super().reset(state)

        self._weighted_tardy_jobs = 0.0

    @override
    def on_task_completed(
        self,
        task_id: TaskID,
        machine_id: MachineID,
        state: ScheduleState,
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        already_tardy = self._tardy_jobs[job_id]

        super().on_task_completed(task_id, machine_id, state)

        if already_tardy:
            return

        if self._tardy_jobs[job_id]:
            self._weighted_tardy_jobs += self.weights.value[job_id]

    @override
    def get_current(self, state: ScheduleState) -> float:
        return self._weighted_tardy_jobs

    @override
    def __call__(self, state: ScheduleState) -> float:
        return sum(
            w_j
            for w_j, d_j, C_j in zip(
                self.weights.value,
                self.due_dates.value,
                self.completion_times(state),
                strict=False,
            )
            if C_j > d_j
        )

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Σw_jU_j"

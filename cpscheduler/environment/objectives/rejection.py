"""Rejection cost objective."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import Float, TaskID
from cpscheduler.environment.instance import Feature, ProblemInstance
from cpscheduler.environment.objectives.base import Objective
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.backtrack import GenericTrail
from cpscheduler.environment.utils.general import convert_to_list


class RejectionCost(Objective):
    """Rejection Cost objective.

    This objective function charges a cost e_j for every rejected job, i.e.,
    Σe_j over the rejected jobs, where a job is rejected when all its tasks are
    absent. It is usually composed with another objective for scheduling with
    rejection, where leaving a job out of the schedule is allowed at a price
    (Shabtay et al., 2013). See `RejectableConstraint` to reject whole jobs.
    """

    rejection_costs: Feature[list[float]]

    _value: float
    _open: float

    _n_present: list[int]
    _n_absent: list[int]

    _trail: GenericTrail

    def __init__(
        self,
        rejection_tag: str = "rejection_cost",
        rejection_costs: Iterable[Float] | None = None,
    ) -> None:
        """Initialize the Rejection Cost objective.

        Parameters
        ----------
        rejection_tag: str, optional
            The name of the job feature that contains the rejection costs.
            Default to "rejection_cost".

        rejection_costs: Iterable[Float] | None, optional
            The rejection cost of each job.
            If None is provided, the costs will be loaded from the instance.
            Default to None.

        """
        super().__init__()

        self.rejection_costs = Feature(
            name=rejection_tag,
            shape=("n_jobs",),
            value_type="cost",
            preprocess=self._load_costs,
        )

        if rejection_costs is not None:
            self.rejection_costs.own_data(rejection_costs)

    def _load_costs(self, costs: Iterable[Float]) -> list[float]:
        return convert_to_list(costs, float)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        super().initialize(instance)

        if any(cost < 0 for cost in self.rejection_costs.value):
            raise ValueError("Rejection costs must be non-negative.")

    @property
    @override
    def backtrack_safe(self) -> bool:
        return True

    @property
    @override
    def regular(self) -> bool:
        return True

    @property
    @override
    def anti_regular(self) -> bool:
        return True

    @property
    @override
    def lb(self) -> float:
        return self._value

    @property
    @override
    def ub(self) -> float:
        return self._value + self._open

    @property
    @override
    def value(self) -> float:
        return self._value

    @override
    def compute(self, state: ScheduleState) -> float:
        costs = self.rejection_costs.value

        return float(
            sum(
                costs[job_id]
                for job_id, tasks in enumerate(state.instance.job_tasks)
                if tasks and all(state.is_absent(task_id) for task_id in tasks)
            )
        )

    @override
    def reset(self, state: ScheduleState) -> None:
        # The state may already be ahead of its initial domains, as the
        # constraints are reset first, but their changes are still delivered
        # as events. So the counts start from the instance, not the state.
        instance = state.instance

        self._trail = GenericTrail(
            "_value", "_open", "_n_present", "_n_absent", deep=False
        )
        self._n_present = [
            sum(not instance.optional[task_id] for task_id in tasks)
            for tasks in instance.job_tasks
        ]
        self._n_absent = [0] * instance.n_jobs
        self._value = 0.0
        self._open = sum(
            cost
            for cost, tasks, n_present in zip(
                self.rejection_costs.value,
                instance.job_tasks,
                self._n_present,
                strict=True,
            )
            if tasks and n_present == 0
        )

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        job_id = state.get_job_id(task_id)
        self._n_absent[job_id] += 1

        # The last absence rejects the job, which had no present task
        if self._n_absent[job_id] == len(state.instance.job_tasks[job_id]):
            cost = self.rejection_costs.value[job_id]
            self._value += cost
            self._open -= cost

    @override
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        job_id = state.get_job_id(task_id)
        self._n_present[job_id] += 1

        # The first presence keeps the job in the schedule
        if self._n_present[job_id] == 1:
            self._open -= self.rejection_costs.value[job_id]

    @override
    def checkpoint(self, mark: int) -> None:
        self._trail.checkpoint(mark, self)

    @override
    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        self._trail.backtrack(mark, self)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Σe_j"

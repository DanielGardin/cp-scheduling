"""Base class for objective functions in the scheduling environment."""

from heapq import heappop, heappush
from typing import ClassVar, Final, Literal

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.component import Component
from cpscheduler.environment.constants import (
    JobID,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.backtrack import GenericTrail

Sense = Literal["minimize", "maximize"]
"The direction of an optimization problem."

objectives: dict[str, type["Objective"]] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Objective(Component):
    """Base class for all objective functions in the scheduling environment.

    Objective functions are used to evaluate the performance of a scheduling
    algorithm.
    They can be used to guide the search for an optimal schedule by providing a
    numerical value that represents the quality of the schedule.
    """

    default_sense: ClassVar[Sense] = "minimize"
    """The sense an environment uses when none is given explicitly.

    The objective is only a function of the schedule: the actual sense of the
    optimization problem belongs to the environment (`SchedulingEnv.sense`).
    """

    @override
    def __init_subclass__(cls) -> None:
        name = cls.__name__

        if not name.startswith("_"):
            objectives[name] = cls

    @property
    def backtrack_safe(self) -> bool:
        """Whether `checkpoint` and `backtrack` restore the objective exactly."""
        return False

    @property
    def regular(self) -> bool:
        r"""The objective is regular, when it is non-decreasing w.r.t completion times.

        That is, an objective f is regular if, whenever $C_i \leq C_i'$,

        \[f(C_1, \cdots, C_n) \leq f(C_1', \cdots, C_n').\]
        """
        return False

    @property
    def anti_regular(self) -> bool:
        r"""The objective is anti-regular, when it is non-increasing w.r.t completion times.

        That is, an objective f is anti-regular if, whenever $C_i \leq C_i'$,

        \[f(C_1, \cdots, C_n) \geq f(C_1', \cdots, C_n').\]
        """
        return False

    @property
    def lb(self) -> float:
        """Return a lower bound for the current objective value."""
        raise ValueError(
            f"Objective {type(self).__name__} has no current lower bound."
        )

    @property
    def ub(self) -> float:
        """Return a upper bound for the current objective value."""
        raise ValueError(
            f"Objective {type(self).__name__} has no current upper bound."
        )

    @property
    def value(self) -> float:
        """Return the current objective value derived only from fixed tasks."""
        raise ValueError(
            f"Objective {type(self).__name__} has no current value."
        )

    def __repr__(self) -> str:
        """Return a string representation of the objective function."""
        return f"{type(self).__name__}()"

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the event of a task being assigned to a machine."""

    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the event of a task's start time lower bound being updated."""

    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the event of a task's start time upper bound being updated."""

    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the event of a task's end time lower bound being updated."""

    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the event of a task's end time upper bound being updated."""

    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle the event of a task's presence being updated."""

    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle the event of a task's absence being updated."""

    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the event of a task being marked as infeasible on a machine."""

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        """Handle the event of the current time being updated."""

    def on_task_lock(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle the event of a task being locked."""

    def on_task_unlock(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle the event of a task being unlocked."""

    def on_failure(self, state: ScheduleState) -> None:
        """Handle the schedule state becoming infeasible."""

    def checkpoint(self, mark: int) -> None:
        """Checkpoints the current state for backtracking."""

    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        """Recompute the objective value when backtracking."""
        raise NotImplementedError(
            f"Objective {type(self).__name__} has not implemented backtrack(); "
            "either implement it or leave backtrack_safe as False."
        )

    def compute(self, state: ScheduleState) -> float:
        """Cold computation of the realized objective value."""
        raise NotImplementedError(
            f"Objective {type(self).__name__} has no implementation of "
            "the compute method."
        )

    def __call__(self, state: ScheduleState) -> float:
        """Return the current realized value, default value when used as a Metric."""
        return self.compute(state)


class SatisfactionObjective(Objective):
    """Satisfaction problem.

    A satisfaction problem is a scheduling problem where the goal is finding
    a feasible schedule.
    This objective takes value of 1 whenever the current schedule is feasible
    and complete, and 0 otherwise.

    This is the default objective when none is passed explicitly to the
    environment.
    It is maximized by default, a feasible schedule is the goal.
    """

    _value: float
    _lb: float
    _ub: float

    default_sense: ClassVar[Sense] = "maximize"

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
    def lb(self) -> float:
        return self._lb

    @property
    @override
    def ub(self) -> float:
        return self._ub

    @property
    @override
    def value(self) -> float:
        return self._value

    @override
    def reset(self, state: "ScheduleState") -> None:
        self._value = 0.0
        self._lb = 0.0
        self._ub = 1.0

    def compute(self, state: ScheduleState) -> float:
        """Cold computation of the realized objective value."""
        return float(state.remaining_tasks == 0 and not state.infeasible)

    def _complete(self, state: ScheduleState) -> None:
        if state.is_terminal() and not state.infeasible:
            self._value = 1.0
            self._lb = 1.0

    @override
    def on_failure(self, state: ScheduleState) -> None:
        self._value = 0.0
        self._lb = 0.0
        self._ub = 0.0

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._complete(state)

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        self._complete(state)

    @override
    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        value = (
            1.0 if state.remaining_tasks == 0 and not state.infeasible else 0.0
        )
        self._value = value
        self._lb = 0.0 if state.infeasible else value
        self._ub = 0.0 if state.infeasible else 1.0

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "—"


_NO_TIME: Final[Time] = -1
"Sentinel for a job without tasks contributing to a completion."

_NO_COST: Final[float] = float("-inf")
"Sentinel for a maximum over an empty set of tasks."


def completion_times(state: ScheduleState) -> list[Time]:
    """Return the completion of every job from its assigned tasks.

    Jobs without assigned tasks have no completion, marked by a negative time.
    """
    completions: list[Time] = [_NO_TIME] * state.n_jobs

    for task_id in state.get_assigned_tasks():
        job_id = state.get_job_id(task_id)
        completions[job_id] = max(completions[job_id], state.get_end(task_id))

    return completions


class _CompletionObjective(Objective):
    """Base class for objectives f(C_1, ..., C_n) aggregating job costs g_j(C_j).

    The per-job costs are declared by `job_cost`, and must be monotone in the
    completion time, in the direction given by `increasing`. The value and
    bounds are maintained incrementally from the domain events, each event
    refreshing the bounds of the task's job.

    A job without present tasks has no completion and costs nothing, so a
    rejected job never changes the objective.

    No reference to the schedule state is kept: every handler receives it.
    """

    increasing: ClassVar[bool] = True
    "Whether the job costs are non-decreasing in the completion time."

    _trailed: ClassVar[tuple[str, ...]] = ()
    "The fields restored on backtrack, replaced but never mutated in place."

    _trail: GenericTrail

    def job_cost(self, job_id: JobID, completion: Time) -> float:
        """Return g_j(C_j), the cost of a job finishing at `completion`."""
        raise NotImplementedError(
            f"Objective {type(self).__name__} has no implementation of "
            "the `job_cost` method."
        )

    def _aggregate(self, costs: list[float]) -> float:
        raise NotImplementedError

    def evaluate(self, job_completions: list[Time]) -> float:
        """Evaluate f(C_1, ..., C_n) from the completion of every job.

        A negative completion marks a job without completion, such as a
        rejected one, which contributes nothing.
        """
        return self._aggregate(
            [
                self.job_cost(job_id, completion)
                for job_id, completion in enumerate(job_completions)
                if completion >= 0
            ]
        )

    @override
    def compute(self, state: ScheduleState) -> float:
        return self.evaluate(completion_times(state))

    @property
    @override
    def backtrack_safe(self) -> bool:
        return True

    @property
    @override
    def regular(self) -> bool:
        return self.increasing

    @property
    @override
    def anti_regular(self) -> bool:
        return not self.increasing

    def _rebuild(self, state: ScheduleState) -> None:
        """Recompute the value and bounds from the domains."""
        raise NotImplementedError

    def _lower_bound_event(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle the end lower bound of a task, or its presence, changing."""

    def _upper_bound_event(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle the end upper bound of a task, or its absence, changing."""

    def _value_event(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle a task being assigned."""

    @override
    def reset(self, state: ScheduleState) -> None:
        self._trail = GenericTrail(*self._trailed, deep=False)
        self._rebuild(state)

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._lower_bound_event(task_id, state)
        self._upper_bound_event(task_id, state)
        self._value_event(task_id, state)

    @override
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        self._lower_bound_event(task_id, state)

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        self._upper_bound_event(task_id, state)

    @override
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._lower_bound_event(task_id, state)

    @override
    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._lower_bound_event(task_id, state)

    @override
    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._upper_bound_event(task_id, state)

    @override
    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._upper_bound_event(task_id, state)

    @override
    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        # Removing a machine moves both global bounds without bound events
        self._lower_bound_event(task_id, state)
        self._upper_bound_event(task_id, state)

    @override
    def checkpoint(self, mark: int) -> None:
        self._trail.checkpoint(mark, self)

    @override
    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        self._trail.backtrack(mark, self)


class _SumCompletionObjective(_CompletionObjective):
    """Base class for objectives f = sum_j g_j(C_j).

    Every job keeps the bounds of its completion and its partial completion,
    and its contribution to the lower bound, upper bound and value. A change
    to a job replaces its contributions in the sums, so every read is O(1).

    Lower bounds only grow during propagation, and are updated in O(1). Upper
    bounds can drop with any task of the job, and are recomputed in
    O(tasks in the job). Updates are idempotent, as events produced before
    `reset` are delivered after it.
    """

    _job_lb: list[Time]
    "Per job, the latest end lower bound of its present tasks."

    _job_ub: list[Time]
    "Per job, the latest end upper bound of the tasks that are not absent."

    _job_partial: list[Time]
    "Per job, the latest end of its assigned tasks."

    _terms: list[tuple[float, float, float]]
    "Per job, its contribution to the lower bound, upper bound and value."

    _lb_cost: float
    _ub_cost: float
    _value_cost: float

    _trailed = (
        "_job_lb",
        "_job_ub",
        "_job_partial",
        "_terms",
        "_lb_cost",
        "_ub_cost",
        "_value_cost",
    )

    @override
    def _aggregate(self, costs: list[float]) -> float:
        return sum(costs)

    @property
    @override
    def lb(self) -> float:
        return self._lb_cost

    @property
    @override
    def ub(self) -> float:
        return self._ub_cost

    @property
    @override
    def value(self) -> float:
        return self._value_cost

    def _refresh(self, job_id: JobID) -> None:
        """Replace the contributions of a job, from its completion times."""
        lb = self._job_lb[job_id]
        ub = self._job_ub[job_id]
        partial = self._job_partial[job_id]

        if ub < 0:  # rejected
            low = high = 0.0

        elif lb >= 0:  # surely present
            early, late = (lb, ub) if self.increasing else (ub, lb)
            low = self.job_cost(job_id, early)
            high = self.job_cost(job_id, late)

        else:  # it may be rejected, or complete in [0, ub]
            costs = (0.0, self.job_cost(job_id, 0), self.job_cost(job_id, ub))
            low, high = min(costs), max(costs)

        value = self.job_cost(job_id, partial) if partial >= 0 else 0.0

        old_low, old_high, old_value = self._terms[job_id]
        self._lb_cost += low - old_low
        self._ub_cost += high - old_high
        self._value_cost += value - old_value
        self._terms[job_id] = (low, high, value)

    @override
    def _rebuild(self, state: ScheduleState) -> None:
        n_jobs = state.n_jobs

        self._job_lb = [_NO_TIME] * n_jobs
        self._job_ub = [_NO_TIME] * n_jobs
        self._job_partial = [_NO_TIME] * n_jobs

        for task_id in range(state.n_tasks):
            if state.is_absent(task_id):
                continue

            job_id = state.get_job_id(task_id)
            self._job_ub[job_id] = max(
                self._job_ub[job_id], state.get_end_ub(task_id)
            )

            if state.is_present(task_id):
                self._job_lb[job_id] = max(
                    self._job_lb[job_id], state.get_end_lb(task_id)
                )

            if state.is_assigned(task_id):
                self._job_partial[job_id] = max(
                    self._job_partial[job_id], state.get_end(task_id)
                )

        self._terms = [(0.0, 0.0, 0.0)] * n_jobs
        self._lb_cost = self._ub_cost = self._value_cost = 0.0

        for job_id in range(n_jobs):
            self._refresh(job_id)

    @override
    def _lower_bound_event(self, task_id: TaskID, state: ScheduleState) -> None:
        if not state.is_present(task_id):
            return

        job_id = state.get_job_id(task_id)
        end_lb = state.get_end_lb(task_id)

        if end_lb > self._job_lb[job_id]:
            self._job_lb[job_id] = end_lb
            self._refresh(job_id)

    @override
    def _upper_bound_event(self, task_id: TaskID, state: ScheduleState) -> None:
        job_id = state.get_job_id(task_id)

        ub = max(
            (
                state.get_end_ub(other_id)
                for other_id in state.instance.job_tasks[job_id]
                if not state.is_absent(other_id)
            ),
            default=_NO_TIME,
        )

        if ub != self._job_ub[job_id]:
            self._job_ub[job_id] = ub
            self._refresh(job_id)

    @override
    def _value_event(self, task_id: TaskID, state: ScheduleState) -> None:
        job_id = state.get_job_id(task_id)
        end = state.get_end(task_id)

        if end > self._job_partial[job_id]:
            self._job_partial[job_id] = end
            self._refresh(job_id)


class _RegularObjective(_SumCompletionObjective):
    """Base class for regular sum objectives.

    An objective is called regular when it is represented by a non-decreasing
    function f(C_1, ..., C_n), where C_i is job i's completion times.
    """


RegularObjective = _RegularObjective


class _AntiRegularObjective(_SumCompletionObjective):
    """Base class for sum objectives that improve when jobs complete later.

    The value is a non-increasing function of the completion times, so its lower
    bound comes from the latest possible completions, and vice versa.
    """

    increasing = False


AntiRegularObjective = _AntiRegularObjective


class _MaxCompletionObjective(_CompletionObjective):
    """Base class for objectives f = max_j g_j(C_j), with non-decreasing g_j.

    Since g_j is non-decreasing, the cost of a job is the largest cost of its
    tasks, g_j(max_t C_t) = max_t g_j(C_t), so the objective is maintained per
    task, without grouping tasks into jobs. The lower bound and value only grow
    during propagation, and are running maxima. The upper bound can drop, and
    is kept in a lazy max-heap validated against each task's end upper bound.

    The maximum over no task is 0.
    """

    _task_ub: list[Time]
    "Per task, its end upper bound, or _NO_TIME if absent."

    _ub_heap: list[tuple[float, TaskID, Time]]
    "Lazy max-heap of (-cost, task, end upper bound)."

    _lb_cost: float
    _value_cost: float

    _trailed = ("_task_ub", "_ub_heap", "_lb_cost", "_value_cost")

    @override
    def _aggregate(self, costs: list[float]) -> float:
        return max(costs, default=0.0)

    @property
    @override
    def lb(self) -> float:
        return self._lb_cost if self._lb_cost != _NO_COST else 0.0

    @property
    @override
    def ub(self) -> float:
        heap = self._ub_heap

        while heap:
            cost, task_id, ub = heap[0]
            if ub == self._task_ub[task_id]:
                return -cost

            heappop(heap)

        return 0.0

    @property
    @override
    def value(self) -> float:
        return self._value_cost if self._value_cost != _NO_COST else 0.0

    def _task_cost(
        self, task_id: TaskID, end: Time, state: ScheduleState
    ) -> float:
        return self.job_cost(state.get_job_id(task_id), end)

    @override
    def _rebuild(self, state: ScheduleState) -> None:
        self._task_ub = [_NO_TIME] * state.n_tasks
        self._ub_heap = []
        self._lb_cost = _NO_COST
        self._value_cost = _NO_COST

        for task_id in range(state.n_tasks):
            self._lower_bound_event(task_id, state)
            self._upper_bound_event(task_id, state)

            if state.is_assigned(task_id):
                self._value_event(task_id, state)

    @override
    def _lower_bound_event(self, task_id: TaskID, state: ScheduleState) -> None:
        if state.is_present(task_id):
            cost = self._task_cost(task_id, state.get_end_lb(task_id), state)
            self._lb_cost = max(self._lb_cost, cost)

    @override
    def _upper_bound_event(self, task_id: TaskID, state: ScheduleState) -> None:
        ub = _NO_TIME if state.is_absent(task_id) else state.get_end_ub(task_id)

        if ub == self._task_ub[task_id]:
            return

        self._task_ub[task_id] = ub
        if ub >= 0:
            cost = self._task_cost(task_id, ub, state)
            heappush(self._ub_heap, (-cost, task_id, ub))

    @override
    def _value_event(self, task_id: TaskID, state: ScheduleState) -> None:
        cost = self._task_cost(task_id, state.get_end(task_id), state)
        self._value_cost = max(self._value_cost, cost)


MaxCompletionObjective = _MaxCompletionObjective

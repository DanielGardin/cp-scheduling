"""Base class for objective functions in the scheduling environment."""

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.component import Component
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state import ScheduleState

objectives: dict[str, type["Objective"]] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Objective(Component):
    """Base class for all objective functions in the scheduling environment.

    Objective functions are used to evaluate the performance of a scheduling
    algorithm.
    They can be used to guide the search for an optimal schedule by providing a
    numerical value that represents the quality of the schedule.
    """

    minimize: bool
    _lb: float
    _ub: float
    _current: float

    @override
    def __init_subclass__(cls) -> None:
        name = cls.__name__

        if not name.startswith("_"):
            objectives[name] = cls

    def __init__(self, minimize: bool = True) -> None:
        """Initialize the Objective.

        Parameters
        ----------
        minimize: bool
            Whether the objective should be minimized (True) or maximized (False).

        """
        self.minimize = minimize
        self._lb = float("-inf")
        self._ub = float("inf")
        self._current = float("-inf")

    @property
    def regular(self) -> bool:
        r"""The objective is regular, when it is non-decreasing w.r.t completion times.

        That is, an objective f is regular (minimization) if, whenever $C_i \leq C_i'$,

        \[f(C_1, \cdots, C_n) \leq f(C_1', \cdots, C_n').\]
        """
        return False

    @property
    def lb(self) -> float:
        """Return a lower bound for the current objective value."""
        return self._lb

    @property
    def ub(self) -> float:
        """Return a upper bound for the current objective value."""
        return self._ub

    @property
    def current(self) -> float:
        """Return the current objective value derived only from fixed tasks."""
        return self._current

    def __repr__(self) -> str:
        """Return a string representation of the objective function."""
        sense = "minimize" if self.minimize else "maximize"

        return f"{type(self).__name__}(sense={sense})"

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

    def compute(self, state: ScheduleState) -> float:
        """Cold computation of the realized objective value."""
        raise NotImplementedError(
            f"Objective {type(self).__name__} has no implementation of "
            "the compute method."
        )

    def __call__(self, state: ScheduleState) -> float:
        """Return the current realized value, default value when used as a Metric."""
        return self.compute(state)


class _CompletionTimeObjective(Objective):
    """Base class for objectives that depend on job completion times.

    This class provides a common implementation for tracking job completion times and
    computing the objective value based on them, not specific to any particular objective.
    """

    _job_completion: list[Time]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        self._job_completion = [0] * instance.n_jobs

    @override
    def reset(self, state: ScheduleState) -> None:
        self._job_completion[:] = [0] * state.n_jobs

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        C_j = state.get_end(task_id)
        self._job_completion[job_id] = max(self._job_completion[job_id], C_j)

    @staticmethod
    def completion_times(state: ScheduleState) -> list[Time]:
        """Compute the makespan of a set of tasks."""
        makespans: list[Time] = [0] * state.n_jobs

        job_ids = state.instance.job_ids

        for task_id in state.get_assigned_tasks():
            job_id = job_ids[task_id]
            C_j = state.get_end(task_id)

            makespans[job_id] = max(makespans[job_id], C_j)

        return makespans


CompletionTimeObjective = _CompletionTimeObjective


class _RegularObjective(CompletionTimeObjective):
    """Base class for regular objectives that depend on job completion times.

    A regular objective is one that is non-decreasing with respect to the
    completion times of the jobs.
    They are a common class of objectives in scheduling problems due to their
    desirable properties for optimization and analysis.

    This base class provides a common implementation for regular objectives
    that depend on job completion times.
    """

    @property
    @override
    def regular(self) -> bool:
        return True


RegularObjective = _RegularObjective

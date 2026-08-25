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


class SatisfactionObjective(Objective):
    """Satisfaction problem.

    A satisfaction problem is a scheduling problem where the goal is finding
    a feasible schedule.
    This objective takes value of 1 whenever the current schedule is feasible
    and complete, and 0 otherwise.

    This is the default objective when none is passed explicitly to the
    environment.
    """

    _value: float
    _lb: float
    _ub: float

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

    @override
    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if state.infeasible:
            self._ub = 0.0

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if state.remaining_tasks == 0 and not state.infeasible:
            self._value = 1.0
            self._lb = 1.0

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "—"


def completion_times(state: ScheduleState) -> list[Time]:
    """Compute the makespan of a set of tasks."""
    makespans: list[Time] = [0] * state.n_jobs

    job_ids = state.instance.job_ids

    for task_id in state.get_assigned_tasks():
        job_id = job_ids[task_id]
        C_j = state.get_end(task_id)

        makespans[job_id] = max(makespans[job_id], C_j)

    return makespans


class _CompletionTimeObjective(Objective):
    """Util class for objectives that depend on job completion times."""

    _job_completion: list[Time]
    _job_completion_lb: list[Time]
    _job_completion_ub: list[Time]

    def initialize(self, instance: ProblemInstance) -> None:
        self._job_completion = [0] * instance.n_jobs
        self._job_completion_lb = [0] * instance.n_jobs
        self._job_completion_ub = [0] * instance.n_jobs

    @override
    def reset(self, state: ScheduleState) -> None:
        self._job_completion[:] = [0] * state.n_jobs
        self._job_completion_lb[:] = [0] * state.n_jobs
        self._job_completion_ub[:] = [0] * state.n_jobs

    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        end_lb = state.get_end_lb(task_id)
        self._job_completion[job_id] = max(self._job_completion[job_id], end_lb)

    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        end_lb = state.get_end_lb(task_id)
        self._job_completion_lb[job_id] = max(
            self._job_completion_lb[job_id], end_lb
        )

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        C_j = state.get_end(task_id)
        self._job_completion_lb[job_id] = max(
            self._job_completion_lb[job_id], C_j
        )

    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        end_ub = state.get_end_ub(task_id)
        self._job_completion_ub[job_id] = max(
            self._job_completion_ub[job_id], end_ub
        )

    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.get_job_id(task_id)
        end_ub = state.get_end_ub(task_id)
        self._job_completion_ub[job_id] = max(
            self._job_completion_ub[job_id], end_ub
        )

    def compute(self, state: ScheduleState) -> float:
        return self.evaluate(completion_times(state))

    def evaluate(self, job_completions: list[Time]) -> float:
        """Evaluate f(C_1, ..., C_n)."""
        raise NotImplementedError(
            f"Objective {type(self).__name__} has no implementation of "
            "the `evaluate` method."
        )


class _RegularObjective(_CompletionTimeObjective):
    """Base class for regular objectives.

    An objective is called regular when it is represented by a non-decreasing
    function f(C_1, ..., C_n), where C_i is job i's completion times,
    """

    @property
    @override
    def regular(self) -> bool:
        return True

    @property
    def lb(self) -> float:
        return self.evaluate(self._job_completion_lb)

    @property
    def ub(self) -> float:
        return self.evaluate(self._job_completion_ub)

    @property
    def value(self) -> float:
        """Return the current objective value derived only from fixed tasks."""
        return self.evaluate(self._job_completion)


RegularObjective = _RegularObjective


class _AntiRegularObjective(_CompletionTimeObjective):
    """Base class for objectives that depend on job completion times.

    To compute the objective value, just implement the `evaluate` method that
    takes completion time estimates per job and return a single value.
    Note that the values computed here are only valid when the objective is
    regular, that is, the `evaluate` function is element-wise non-decreasing.
    """

    @property
    @override
    def regular(self) -> bool:
        return False

    @property
    def lb(self) -> float:
        return self.evaluate(self._job_completion_ub)

    @property
    def ub(self) -> float:
        return self.evaluate(self._job_completion_lb)

    @property
    def value(self) -> float:
        """Return the current objective value derived only from fixed tasks."""
        return self.evaluate(self._job_completion)


AntiRegularObjective = _AntiRegularObjective

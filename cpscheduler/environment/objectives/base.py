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
        """The objective is regular, when it is non-decreasing w.r.t completion times."""
        return False

    def __repr__(self) -> str:
        """Return a string representation of the objective function."""
        sense = "minimize" if self.minimize else "maximize"

        return f"{type(self).__name__}(sense={sense})"

    def on_task_started(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the objective change when a task starts its execution."""

    def on_task_paused(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the objective change when a task is interrupted."""

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the objective change when a task ends its execution."""

    def on_task_machine_infeasible(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the objective change when a task loses feasibility on a machine."""

    def get_current(self, state: ScheduleState) -> float:
        """Get the current value of the objective function.

        This is used for retrieving the objective value during the episode.
        """
        return 0.0

    def __call__(self, state: ScheduleState) -> float:
        """Call the objective function to get the current value."""
        return self.get_current(state)


class CompletionTimeObjective(Objective):
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
    def on_task_completed(
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

        for task_id in state.get_completed_tasks():
            job_id = job_ids[task_id]
            C_j = state.get_end(task_id)

            makespans[job_id] = max(makespans[job_id], C_j)

        return makespans


class RegularObjective(CompletionTimeObjective):
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

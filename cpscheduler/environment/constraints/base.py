"""Base constraint class for scheduling environments."""

from typing import ClassVar, NoReturn, final

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.component import Component
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.state import ScheduleState

constraints: dict[str, type["Constraint"]] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Constraint(Component):
    """Base class for all constraints in the scheduling environment.

    This class provides a common interface for any piece in the scheduling
    environment that interacts with the tasks by limiting when they can be
    executed, how they are assigned to machines, etc.

    If the propagator is stateless and pure, with only event-creating calls
    inside the propagation functions, then the constraint is safe for backtracking,
    and the class should have backtrack_safe = True explicitly.

    When backtrack_safe = False, the environment cannot backtrack, because
    there is no guarantee that returning to a checkpoint will load the correct
    constraint state, and not a stale one.
    """

    backtrack_safe: ClassVar[bool] = False

    @override
    def __init_subclass__(cls) -> None:
        name = cls.__name__

        if not name.startswith("_"):
            constraints[name] = cls

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

    def checkpoint(self, mark: int) -> None:
        """Checkpoints the current state for backtracking."""

    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        """Restore the inner state when backtracking."""


# FUTURE: Remove Passive constraint class when implementing the subscription
# feature during propagation.
class PassiveConstraint(Constraint):
    """Compile-time constraints that only interact with the instance at initialization.

    They are used to provide task information and to set up the initial state
    for the scheduler.
    """

    backtrack_safe = True

    @final
    def reset(self, state: ScheduleState) -> NoReturn:
        """Passive constraint does not reset any state."""
        raise RuntimeError("Passive constraint does not reset any state.")

    @final
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        """Passive constraint does not handle assignment events."""
        raise RuntimeError(
            "Passive constraint does not handle assignment events."
        )

    @final
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        """Passive constraint does not handle start time lower bound events."""
        raise RuntimeError(
            "Passive constraint does not handle start time lower bound events."
        )

    @final
    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        """Passive constraint does not handle start time upper bound events."""
        raise RuntimeError(
            "Passive constraint does not handle start time upper bound events."
        )

    @final
    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        """Passive constraint does not handle end time lower bound events."""
        raise RuntimeError(
            "Passive constraint does not handle end time lower bound events."
        )

    @final
    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        """Passive constraint does not handle end time upper bound events."""
        raise RuntimeError(
            "Passive constraint does not handle end time upper bound events."
        )

    @final
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> NoReturn:
        """Passive constraint does not handle presence events."""
        raise RuntimeError(
            "Passive constraint does not handle presence events."
        )

    @final
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> NoReturn:
        """Passive constraint does not handle absence events."""
        raise RuntimeError("Passive constraint does not handle absence events.")

    @final
    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        """Passive constraint does not handle infeasibility events."""
        raise RuntimeError(
            "Passive constraint does not handle infeasibility events."
        )

    @final
    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> NoReturn:
        """Passive constraint does not participate in backtracking."""
        raise RuntimeError(
            "Passive constraint does not participate in backtracking."
        )

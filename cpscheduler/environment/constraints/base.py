"""Base constraint class for scheduling environments."""

from typing import NoReturn, final, override

from mypy_extensions import mypyc_attr

from cpscheduler.environment.component import Component
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.state.events import VarField

ASSIGNMENT = VarField.ASSIGNMENT
START_LB = VarField.START_LB
START_UB = VarField.START_UB
END_LB = VarField.END_LB
END_UB = VarField.END_UB
PRESENCE = VarField.PRESENCE
ABSENCE = VarField.ABSENCE
MACHINE_INFEASIBLE = VarField.MACHINE_INFEASIBLE
PAUSE = VarField.PAUSE
BOUNDS_RESET = VarField.BOUNDS_RESET
STATE_INFEASIBLE = VarField.STATE_INFEASIBLE

constraints: dict[str, type["Constraint"]] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Constraint(Component):
    """Base class for all constraints in the scheduling environment.

    This class provides a common interface for any piece in the scheduling
    environment that interacts with the tasks by limiting when they can be
    executed, how they are assigned to machines, etc.
    """

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

    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the invalidation of bounds of a task that was paused."""

    def on_bound_reset(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle the bound invalidation of a given task."""

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        """Handle the event of the current time being updated."""


# FUTURE: Remove Passive constraint class when implementing the subscription
# feature during propagation.
class PassiveConstraint(Constraint):
    """Compile-time constraints that only interact with the instance at initialization.

    They are used to provide task information and to set up the initial state
    for the scheduler.
    """

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
    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle the invalidation of bounds of a task that was paused."""

    @final
    def on_bound_reset(self, task_id: TaskID, state: ScheduleState) -> None:
        """Handle the bound invalidation of a given task."""

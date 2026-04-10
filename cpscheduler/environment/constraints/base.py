from typing import Any, NoReturn, final
from mypy_extensions import mypyc_attr

from cpscheduler.environment.state.events import DomainEvent, VarField
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.constants import TaskID, MachineID, Time

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


@mypyc_attr(allow_interpreted_subclasses=True)
class Constraint:
    """
    Base class for all constraints in the scheduling environment.
    This class provides a common interface for any piece in the scheduling environment that
    interacts with the tasks by limiting when they can be executed, how they are assigned to
    machines, etc.
    """

    def __init_subclass__(cls) -> None:
        constraints[cls.__name__] = cls

    def initialize(self, state: ScheduleState) -> None:
        """
        Initialize the constraint with the scheduling state.

        This operation is meant to initialize the internal state of the constraint given the
        observed state at the constraint's inclusion time.
        """

    def get_observation(self) -> dict[str, list[Any]]:
        "Export"
        return {}

    def reset(self, state: ScheduleState) -> None:
        "Reset the constraint to its initial state."

    @final
    def propagate(self, event: DomainEvent, state: ScheduleState) -> None:
        "Deprecated method for propagating a domain event. Use the specific event handlers instead."

        field = event.field
        task_id = event.task_id
        machine_id = event.machine_id

        if field == START_LB:
            self.on_start_lb(task_id, machine_id, state)

        elif field == START_UB:
            self.on_start_ub(task_id, machine_id, state)

        elif field == END_LB:
            self.on_end_lb(task_id, machine_id, state)

        elif field == END_UB:
            self.on_end_ub(task_id, machine_id, state)

        elif field == ASSIGNMENT:
            self.on_assignment(task_id, machine_id, state)

        elif field == PRESENCE:
            self.on_presence(task_id, state)

        elif field == ABSENCE:
            self.on_absence(task_id, state)

        elif field == MACHINE_INFEASIBLE:
            self.on_infeasibility(task_id, machine_id, state)

        elif field == PAUSE:
            self.on_pause(task_id, machine_id, state)

        elif field == BOUNDS_RESET:
            self.on_bound_reset(task_id, state)

        elif field == STATE_INFEASIBLE:
            raise RuntimeError("Cannot propagate in a stale state.")

        else:
            raise ValueError(f"Unknown event field: {field}")

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handle the event of a task being assigned to a machine."

    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handle the event of a task's start time lower bound being updated."

    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handle the event of a task's start time upper bound being updated."

    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handle the event of a task's end time lower bound being updated."

    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handle the event of a task's end time upper bound being updated."

    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        "Handle the event of a task's presence being updated."

    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        "Handle the event of a task's absence being updated."

    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handle the event of a task being marked as infeasible on a machine."

    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handle the invalidation of bounds of a task that was paused."
    
    def on_bound_reset(self, task_id: TaskID, state: ScheduleState) -> None:
        "Handle the bound invalidation of a given task."

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        "Handle the event of the current time being updated."

    def get_entry(self) -> str:
        "Produce the β entry for the constraint."
        return ""

@mypyc_attr(allow_interpreted_subclasses=True)
class PassiveConstraint(Constraint):
    """
    Passive constraints are compile-time constraints on the instance and do not interact with
    events during the scheduling process.
    They are used to provide task information and to set up the initial state for the scheduler.
    """

    @final
    def reset(self, state: ScheduleState) -> NoReturn:
        "Passive constraint does not reset any state."
        raise RuntimeError("Passive constraint does not reset any state.")

    @final
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        "Passive constraint does not handle assignment events."
        raise RuntimeError(
            "Passive constraint does not handle assignment events."
        )

    @final
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        "Passive constraint does not handle start time lower bound events."
        raise RuntimeError(
            "Passive constraint does not handle start time lower bound events."
        )

    @final
    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        "Passive constraint does not handle start time upper bound events."
        raise RuntimeError(
            "Passive constraint does not handle start time upper bound events."
        )

    @final
    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        "Passive constraint does not handle end time lower bound events."
        raise RuntimeError(
            "Passive constraint does not handle end time lower bound events."
        )

    @final
    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        "Passive constraint does not handle end time upper bound events."
        raise RuntimeError(
            "Passive constraint does not handle end time upper bound events."
        )

    @final
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> NoReturn:
        "Passive constraint does not handle presence events."
        raise RuntimeError(
            "Passive constraint does not handle presence events."
        )

    @final
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> NoReturn:
        "Passive constraint does not handle absence events."
        raise RuntimeError("Passive constraint does not handle absence events.")

    @final
    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> NoReturn:
        "Passive constraint does not handle infeasibility events."
        raise RuntimeError(
            "Passive constraint does not handle infeasibility events."
        )

    @final
    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handle the invalidation of bounds of a task that was paused."

    @final
    def on_bound_reset(self, task_id: TaskID, state: ScheduleState) -> None:
        "Handle the bound invalidation of a given task."

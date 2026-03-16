from typing import ClassVar, NoReturn, final
from mypy_extensions import mypyc_attr

from cpscheduler.environment.state.events import DomainEvent, VarField
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.constants import TaskID, MachineID, Time

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
        "Initialize the constraint with the scheduling state."

    def reset(self, state: ScheduleState) -> None:
        "Reset the constraint to its initial state."

    def get_entry(self) -> str:
        "Produce the β entry for the constraint."
        return ""

    @final
    def propagate(self, event: DomainEvent, state: ScheduleState) -> None:
        "Deprecated method for propagating a domain event. Use the specific event handlers instead."
        match event.field:
            case VarField.ASSIGNMENT:
                self.on_assignment(event.task_id, event.machine_id, state)

            case VarField.START_LB:
                self.on_start_lb(event.task_id, event.machine_id, state)

            case VarField.START_UB:
                self.on_start_ub(event.task_id, event.machine_id, state)

            case VarField.END_LB:
                self.on_end_lb(event.task_id, event.machine_id, state)

            case VarField.END_UB:
                self.on_end_ub(event.task_id, event.machine_id, state)

            case VarField.PRESENCE:
                self.on_presence(event.task_id, state)

            case VarField.ABSENCE:
                self.on_absence(event.task_id, state)

            case VarField.INFEASIBLE:
                self.on_infeasibility(event.task_id, event.machine_id, state)

            case _:
                raise ValueError(f"Unknown event field: {event.field}")

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

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        "Handle the event of the current time being updated."


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


class SoftConstraint(Constraint):
    """
    Soft constraints are constraints that can be violated at a cost.
    They are used to model preferences and to provide a more flexible scheduling
    environment.

    They cannot reduce the feasible space of the scheduling problem, only
    record violations to the state via the `record_violation` method.
    """

    violation_name: ClassVar[str]

    def __init_subclass__(cls, violation_name: str | None = None) -> None:
        super().__init_subclass__()

        if violation_name is None:
            violation_name = cls.__name__.lower()

        cls.violation_name = violation_name

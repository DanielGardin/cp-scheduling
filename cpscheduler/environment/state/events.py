from typing import Any, TypeAlias, Final
from mypy_extensions import u8

from cpscheduler.environment.constants import (
    TaskID,
    MachineID,
    GLOBAL_MACHINE_ID,
)

VarFieldType: TypeAlias = u8


class VarField:
    ASSIGNMENT: Final[VarFieldType] = 0
    "A task have its domain colapsed to a single machine and l = u = start time."

    START_LB: Final[VarFieldType] = 1
    "A task have its start interval changed from [l, u) to [l', u), where l' > l."

    START_UB: Final[VarFieldType] = 2
    "A task have its start interval changed from [l, u) to [l, u'), where u' < u."

    END_LB: Final[VarFieldType] = 3
    "A task have its end interval changed from [l, u) to [l', u), where l' > l."

    END_UB: Final[VarFieldType] = 4
    "A task have its end interval changed from [l, u) to [l, u'), where u' < u."

    PRESENCE: Final[VarFieldType] = 5
    "A task have its presence changed from absent to mandatory."

    ABSENCE: Final[VarFieldType] = 6
    "A task have its presence changed from mandatory to absent."

    INFEASIBILITY: Final[VarFieldType] = 7
    "A task has been determined to be infeasible."

    PAUSE: Final[VarFieldType] = 8
    "A task have been paused and its bounds reset."

    BOUNDS_RESET: Final[VarFieldType] = 9
    "A task have its start interval set to [current_time, MAX_INT]."


ASSIGNMENT = VarField.ASSIGNMENT
START_LB = VarField.START_LB
START_UB = VarField.START_UB
END_LB = VarField.END_LB
END_UB = VarField.END_UB
PRESENCE = VarField.PRESENCE
ABSENCE = VarField.ABSENCE
INFEASIBILITY = VarField.INFEASIBILITY
PAUSE = VarField.PAUSE
BOUNDS_RESET = VarField.BOUNDS_RESET

def field_to_str(field: VarFieldType) -> str:
    if field == START_LB:
        return "START_LB"

    elif field == START_UB:
        return "START_UB"

    elif field == END_LB:
        return "END_LB"

    elif field == END_UB:
        return "END_UB"

    elif field == ASSIGNMENT:
        return "ASSIGNMENT"

    elif field == PRESENCE:
        return "PRESENCE"

    elif field == ABSENCE:
        return "ABSENCE"

    elif field == INFEASIBILITY:
        return "INFEASIBILITY"

    elif field == PAUSE:
        return "PAUSE"

    elif field == BOUNDS_RESET:
        return "BOUNDS_RESET"

    else:
        raise ValueError(f"Unknown event field: {field}")


class DomainEvent:
    """
    Base class for CP events in the scheduling environment.
    """

    __slots__ = ("task_id", "field", "machine_id")

    task_id: TaskID
    field: VarFieldType
    machine_id: MachineID

    def __init__(
        self,
        task_id: TaskID,
        field: VarFieldType,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        self.task_id = task_id
        self.field = field
        self.machine_id = machine_id

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            (self.task_id, self.field, self.machine_id),
        )

    def __repr__(self) -> str:
        string = f"DomainEvent(task_id={self.task_id}, field={field_to_str(self.field)}"

        if self.machine_id != GLOBAL_MACHINE_ID:
            string += f", machine_id={self.machine_id}"

        return string + ")"

    def is_assignment(self) -> bool:
        return self.field == ASSIGNMENT

    def is_global(self) -> bool:
        return self.machine_id == GLOBAL_MACHINE_ID

    def is_infeasibility(self) -> bool:
        return (
            self.field == INFEASIBILITY and self.machine_id == GLOBAL_MACHINE_ID
        )

    def is_start_field(self) -> bool:
        field = self.field

        return field == START_LB or field == START_UB

    def is_end_field(self) -> bool:
        field = self.field

        return field == END_LB or field == END_UB

    def is_lower_bound(self) -> bool:
        field = self.field

        return field == START_LB or field == END_LB

    def is_upper_bound(self) -> bool:
        field = self.field

        return field == START_UB or field == END_UB

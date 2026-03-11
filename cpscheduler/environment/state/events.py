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

    INFEASIBLE: Final[VarFieldType] = 7
    "A task has been determined to be infeasible."


class Event:
    """
    Base class for events in the scheduling environment.
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

    def is_assignment(self) -> bool:
        return self.field == VarField.ASSIGNMENT

    def is_global(self) -> bool:
        return self.machine_id == GLOBAL_MACHINE_ID

    def is_infeasibility(self) -> bool:
        return (
            self.field == VarField.INFEASIBLE
            and self.machine_id == GLOBAL_MACHINE_ID
        )

    def is_start_field(self) -> bool:
        field = self.field

        return field == VarField.START_LB or field == VarField.START_UB

    def is_end_field(self) -> bool:
        field = self.field

        return field == VarField.END_LB or field == VarField.END_UB

    def is_lower_bound(self) -> bool:
        field = self.field

        return field == VarField.START_LB or field == VarField.END_LB

    def is_upper_bound(self) -> bool:
        field = self.field

        return field == VarField.START_UB or field == VarField.END_UB

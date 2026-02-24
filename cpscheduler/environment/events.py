from typing import TypeAlias, Final
from mypy_extensions import u8

from cpscheduler.environment._common import TASK_ID, MACHINE_ID, GLOBAL_MACHINE_ID

VarFieldType: TypeAlias = u8

class VarField:
    START_LB: Final[VarFieldType] = 0
    START_UB: Final[VarFieldType] = 1
    END_LB: Final[VarFieldType] = 2
    END_UB: Final[VarFieldType] = 3
    PRESENCE: Final[VarFieldType] = 4

class Event:
    """
    Base class for events in the scheduling environment.
    """
    __slots__ = (
        "task_id",
        "field",
        "machine_id"
    )

    task_id: TASK_ID
    field: VarFieldType
    machine_id: MACHINE_ID

    def __init__(
        self,
        task_id: TASK_ID,
        field: VarFieldType,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> None:
        self.task_id = task_id
        self.field = field
        self.machine_id = machine_id
    
    def is_global(self) -> bool:
        return self.machine_id == GLOBAL_MACHINE_ID

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

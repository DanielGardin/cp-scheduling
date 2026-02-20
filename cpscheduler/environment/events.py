from enum import Enum, auto
from dataclasses import dataclass

from cpscheduler.environment._common import TASK_ID, MACHINE_ID, GLOBAL_MACHINE_ID


class VarField(Enum):
    START_LB = auto()
    START_UB = auto()
    END_LB = auto()
    END_UB = auto()
    PRESENCE = auto()

    def is_start_field(self) -> bool:
        return self == VarField.START_LB or self == VarField.START_UB

    def is_end_field(self) -> bool:
        return self == VarField.END_LB or self == VarField.END_UB

    def is_lower_bound(self) -> bool:
        return self == VarField.START_LB or self == VarField.END_LB

    def is_upper_bound(self) -> bool:
        return self == VarField.START_UB or self == VarField.END_UB


@dataclass
class Event:
    """
    Base class for events in the scheduling environment.
    """

    task_id: TASK_ID
    field: VarField
    machine_id: MACHINE_ID = GLOBAL_MACHINE_ID

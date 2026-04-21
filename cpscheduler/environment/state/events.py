from typing import Final, Literal
from typing_extensions import assert_never

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    TaskID, MachineID,
    GLOBAL_MACHINE_ID,
    Enum, EzPickle
)

VarFieldType = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class VarField(Enum):

    ASSIGNMENT: Final[Literal[0]] = 0
    "A task have its domain colapsed to a single machine and l = u = start time."

    START_LB: Final[Literal[1]] = 1
    "A task have its start interval changed from [l, u) to [l', u), where l' > l."

    START_UB: Final[Literal[2]] = 2
    "A task have its start interval changed from [l, u) to [l, u'), where u' < u."

    END_LB: Final[Literal[3]] = 3
    "A task have its end interval changed from [l, u) to [l', u), where l' > l."

    END_UB: Final[Literal[4]] = 4
    "A task have its end interval changed from [l, u) to [l, u'), where u' < u."

    PRESENCE: Final[Literal[5]] = 5
    "A task have its presence changed from absent to mandatory."

    ABSENCE: Final[Literal[6]] = 6
    "A task have its presence changed from mandatory to absent."

    MACHINE_INFEASIBLE: Final[Literal[7]] = 7
    "A task has been  in some machine."

    PAUSE: Final[Literal[8]] = 8
    "A task have been paused and its bounds reset."

    BOUNDS_RESET: Final[Literal[9]] = 9
    "A task have its start interval set to [current_time, MAX_INT]."

    STATE_INFEASIBLE: Final[Literal[10]] = 10
    "Flag indicating global infeasibility, it is not handled by constraints."

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


def field_to_str(field: VarFieldType) -> str:
    if field == START_LB:
        return "START_LB"

    if field == START_UB:
        return "START_UB"

    if field == END_LB:
        return "END_LB"

    if field == END_UB:
        return "END_UB"

    if field == ASSIGNMENT:
        return "ASSIGNMENT"

    if field == PRESENCE:
        return "PRESENCE"

    if field == ABSENCE:
        return "ABSENCE"

    if field == MACHINE_INFEASIBLE:
        return "MACHINE_INFEASIBLE"

    if field == PAUSE:
        return "PAUSE"

    if field == BOUNDS_RESET:
        return "BOUNDS_RESET"

    if field == STATE_INFEASIBLE:
        return "STATE_INFEASIBLE"

    assert_never(field)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class DomainEvent(EzPickle):
    """
    Container for CP events in the scheduling environment.
    """
    __args__ = ("task_id", "field", "machine_id")

    task_id: TaskID
    field: VarFieldType
    machine_id: MachineID

    def __init__(
        self,
        task_id: TaskID ,
        field: VarFieldType,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        self.task_id = task_id
        self.field = field
        self.machine_id = machine_id

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, DomainEvent):
            return False
        
        return (
            self.task_id == value.task_id
            and self.field == value.field
            and self.machine_id == value.machine_id
        )


EventKindType = Literal[0, 1, 2]

class RuntimeEventKind(Enum):

    TASK_STARTED: Final[Literal[0]] = 0
    TASK_PAUSED: Final[Literal[1]] = 1
    TASK_COMPLETED: Final[Literal[2]] = 2


TASK_STARTED = RuntimeEventKind.TASK_STARTED
TASK_PAUSED = RuntimeEventKind.TASK_PAUSED
TASK_COMPLETED = RuntimeEventKind.TASK_COMPLETED

def kind_to_str(kind: EventKindType) -> str:
    if kind == TASK_STARTED:
        return "TASK_STARTED"

    if kind == TASK_PAUSED:
        return "TASK_PAUSED"
    
    if kind == TASK_COMPLETED:
        return "TASK_COMPLETED"
    
    assert_never(kind)

@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class RuntimeEvent(EzPickle):
    """
    Container for runtime events in the scheduling environment.
    """
    __args__ = ("task_id", "kind", "machine_id")

    task_id: TaskID
    kind: EventKindType
    machine_id: MachineID

    def __init__(
        self,
        task_id: TaskID,
        kind: EventKindType,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        self.task_id = task_id
        self.kind = kind
        self.machine_id = machine_id

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, RuntimeEvent):
            return False
        
        return (
            self.task_id == value.task_id
            and self.kind == value.kind
            and self.machine_id == value.machine_id
        )

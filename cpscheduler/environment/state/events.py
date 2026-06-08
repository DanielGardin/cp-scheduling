"""Event containers and helpers for the scheduling state."""

from typing import Final, Literal

from mypy_extensions import mypyc_attr
from typing_extensions import assert_never

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    Enum,
    EzPickle,
    MachineID,
    TaskID,
)

VarFieldType = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class VarField(Enum):
    """Domain change event kinds used to trigger propagator callbacks."""

    ASSIGNMENT: Final[Literal[0]] = 0
    """A task was committed to a single machine and start time (assignment)."""

    START_LB: Final[Literal[1]] = 1
    """The start lower bound for a task increased."""

    START_UB: Final[Literal[2]] = 2
    """The start upper bound for a task decreased."""

    END_LB: Final[Literal[3]] = 3
    """The end lower bound for a task increased."""

    END_UB: Final[Literal[4]] = 4
    """The end upper bound for a task decreased."""

    PRESENCE: Final[Literal[5]] = 5
    """The presence domain changed to 'present' for a task."""

    ABSENCE: Final[Literal[6]] = 6
    """The presence domain changed to 'absent' for a task."""

    MACHINE_INFEASIBLE: Final[Literal[7]] = 7
    """A machine became infeasible for the given task."""

    PAUSE: Final[Literal[8]] = 8
    """A running task was paused; its remaining time and bounds were updated."""

    BOUNDS_RESET: Final[Literal[9]] = 9
    """Task bounds were reset to the current time horizon."""

    STATE_INFEASIBLE: Final[Literal[10]] = 10
    """Global infeasibility flag signalled by a propagator."""


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
    """Return a short string name for a VarField value.

    Parameters
    ----------
    field : VarFieldType
        Event kind to stringify.

    Returns
    -------
    str
        Uppercase name of the event kind.

    """
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


# FUTURE: Change this event struct to a struct of list of events
@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class DomainEvent(EzPickle):
    """Lightweight value object for CP domain events.

    Attributes
    ----------
    task_id : TaskID
        Identifier of the affected task.

    field : VarFieldType
        Kind of domain change.

    machine_id : MachineID
        Machine identifier associated with the event (or GLOBAL_MACHINE_ID).

    """

    task_id: TaskID
    field: VarFieldType
    machine_id: MachineID

    def __init__(
        self,
        task_id: TaskID,
        field: VarFieldType,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        """Initialize a DomainEvent.

        Parameters
        ----------
        task_id : TaskID
            Identifier of the affected task.

        field : VarFieldType
            Kind of domain change.

        machine_id : MachineID, optional
            Machine identifier associated with the event, or GLOBAL_MACHINE_ID (default).

        """
        self.task_id = task_id
        self.field = field
        self.machine_id = machine_id

    def __eq__(self, value: object, /) -> bool:
        """Check equality of DomainEvent containers."""
        if not isinstance(value, DomainEvent):
            return False

        return (
            self.task_id == value.task_id
            and self.field == value.field
            and self.machine_id == value.machine_id
        )


EventKindType = Literal[0, 1, 2, 3]


class RuntimeEventKind(Enum):
    """Runtime event kinds used to trigger objective and observation callbacks."""

    TASK_STARTED: Final[Literal[0]] = 0
    TASK_PAUSED: Final[Literal[1]] = 1
    TASK_COMPLETED: Final[Literal[2]] = 2
    TASK_MACHINE_INFEASIBLE: Final[Literal[3]] = 3


TASK_STARTED = RuntimeEventKind.TASK_STARTED
TASK_PAUSED = RuntimeEventKind.TASK_PAUSED
TASK_COMPLETED = RuntimeEventKind.TASK_COMPLETED
TASK_MACHINE_INFEASIBLE = RuntimeEventKind.TASK_MACHINE_INFEASIBLE


def kind_to_str(kind: EventKindType) -> str:
    """Return a short string name for a RuntimeEventKind value.

    Parameters
    ----------
    kind : EventKindType
        Runtime event kind to stringify.

    Returns
    -------
    str
        Uppercase name of the runtime event kind.

    """
    if kind == TASK_STARTED:
        return "TASK_STARTED"

    if kind == TASK_PAUSED:
        return "TASK_PAUSED"

    if kind == TASK_COMPLETED:
        return "TASK_COMPLETED"

    if kind == TASK_MACHINE_INFEASIBLE:
        return "TASK_MACHINE_INFEASIBLE"

    assert_never(kind)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class RuntimeEvent(EzPickle):
    """Value object for runtime lifecycle events."""

    task_id: TaskID
    kind: EventKindType
    machine_id: MachineID

    def __init__(
        self,
        task_id: TaskID,
        kind: EventKindType,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        """Initialize a RuntimeEvent.

        Parameters
        ----------
        task_id : TaskID
            Affected task identifier.

        kind : EventKindType
            Specific runtime event (started, paused, completed, machine infeasible).

        machine_id : MachineID, optional
            Machine identifier related to the event, or GLOBAL_MACHINE_ID (default).

        """
        self.task_id = task_id
        self.kind = kind
        self.machine_id = machine_id

    def __eq__(self, value: object, /) -> bool:
        """Check equality of RuntimeEvent containers."""
        return (
            isinstance(value, RuntimeEvent)
            and self.task_id == value.task_id
            and self.kind == value.kind
            and self.machine_id == value.machine_id
        )

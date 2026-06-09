"""Event containers and helpers for the scheduling state."""

from typing import Final, Literal

from mypy_extensions import mypyc_attr

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


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class DomainEventQueue(EzPickle):
    """Container for domain events generated during constraint propagation."""

    task_ids: list[TaskID]
    fields: list[VarFieldType]
    machine_ids: list[MachineID]

    def __init__(self) -> None:
        """Initialize an empty DomainEventQueue."""
        self.task_ids = []
        self.fields = []
        self.machine_ids = []

    def add_event(
        self,
        task_id: TaskID,
        field: VarFieldType,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        """Add a domain event to the queue."""
        self.task_ids.append(task_id)
        self.fields.append(field)
        self.machine_ids.append(machine_id)

    def __len__(self) -> int:
        """Return the number of events in the queue."""
        return len(self.task_ids)

    def clear(self) -> None:
        """Clear all events from the queue."""
        self.task_ids.clear()
        self.fields.clear()
        self.machine_ids.clear()

    def __repr__(self) -> str:
        """Return a string representation of the DomainEventQueue."""
        return f"DomainEventQueue(num_events={len(self.task_ids)})"


EventKindType = Literal[0, 1, 2, 3]


class RuntimeEventKind(Enum):
    """Runtime event kinds used to trigger objective and observation callbacks."""

    TASK_STARTED: Final[Literal[0]] = 0
    TASK_PAUSED: Final[Literal[1]] = 1
    TASK_COMPLETED: Final[Literal[2]] = 2
    TASK_MACHINE_INFEASIBLE: Final[Literal[3]] = 3


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class RuntimeEventQueue(EzPickle):
    """Container for runtime events generated during schedule execution."""

    task_ids: list[TaskID]
    kinds: list[EventKindType]
    machine_ids: list[MachineID]

    def __init__(self) -> None:
        """Initialize an empty RuntimeEventQueue."""
        self.task_ids = []
        self.kinds = []
        self.machine_ids = []

    def add_event(
        self,
        task_id: TaskID,
        kind: EventKindType,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        """Add a runtime event to the queue."""
        self.task_ids.append(task_id)
        self.kinds.append(kind)
        self.machine_ids.append(machine_id)

    def __len__(self) -> int:
        """Return the number of events in the queue."""
        return len(self.task_ids)

    def clear(self) -> None:
        """Clear all events from the queue."""
        self.task_ids.clear()
        self.kinds.clear()
        self.machine_ids.clear()

    def __repr__(self) -> str:
        """Return a string representation of the RuntimeEventQueue."""
        return f"RuntimeEventQueue(num_events={len(self.task_ids)})"

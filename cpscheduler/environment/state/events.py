"""Event containers and helpers for the scheduling state."""

from enum import Enum

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    EzPickle,
    MachineID,
    TaskID,
    Time,
)

TIMELESS: Time = -1


class VarField(Enum):
    """Domain change event kinds used to trigger propagator callbacks."""

    ASSIGNMENT = 0
    """A task was committed to a single machine and start time (assignment)."""

    START_LB = 1
    """The start lower bound for a task increased."""

    START_UB = 2
    """The start upper bound for a task decreased."""

    END_LB = 3
    """The end lower bound for a task increased."""

    END_UB = 4
    """The end upper bound for a task decreased."""

    PRESENCE = 5
    """The presence domain changed to 'present' for a task."""

    ABSENCE = 6
    """The presence domain changed to 'absent' for a task."""

    MACHINE_INFEASIBLE = 7
    """A machine became infeasible for the given task."""

    STATE_INFEASIBLE = 8
    """Global infeasibility flag signalled by a propagator."""

    GLOBAL_TIME = 9
    """All remaining tasks must execute after a global time."""


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False, acyclic=True)
class DomainEventQueue(EzPickle):
    """Container for domain events generated during constraint propagation."""

    task_ids: list[TaskID]
    fields: list[VarField]
    machine_ids: list[MachineID]
    times: list[Time]

    def __init__(self) -> None:
        """Initialize an empty DomainEventQueue."""
        self.task_ids = []
        self.fields = []
        self.machine_ids = []
        self.times = []

    def add_event(
        self,
        task_id: TaskID,
        field: VarField,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
        time: Time = TIMELESS,
    ) -> None:
        """Add a domain event to the queue."""
        self.task_ids.append(task_id)
        self.fields.append(field)
        self.machine_ids.append(machine_id)
        self.times.append(time)

    def __len__(self) -> int:
        """Return the number of events in the queue."""
        return len(self.task_ids)

    def clear(self) -> None:
        """Clear all events from the queue."""
        self.task_ids.clear()
        self.fields.clear()
        self.machine_ids.clear()
        self.times.clear()

    def __repr__(self) -> str:
        """Return a string representation of the DomainEventQueue."""
        return f"DomainEventQueue(num_events={len(self.task_ids)})"

    def __eq__(self, other: object) -> bool:
        """Check equality of two DomainEventQueue instances."""
        return (
            isinstance(other, DomainEventQueue)
            and self.task_ids == other.task_ids
            and self.fields == other.fields
            and self.machine_ids == other.machine_ids
            and self.times == other.times
        )

    def __bool__(self) -> bool:
        """Return True if the queue has any events, False otherwise."""
        return bool(self.task_ids)

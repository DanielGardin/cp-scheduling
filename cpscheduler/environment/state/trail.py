"""Reversible-state trail for backtracking search."""

from enum import IntEnum

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    UNKNOWN_TASK,
    MachineID,
    TaskID,
)
from cpscheduler.environment.mixins import EzPickle


class TrailField(IntEnum):
    """Kind of value recorded in a trail entry."""

    START_LB = 0
    START_UB = 1
    END_LB = 2
    END_UB = 3
    FEASIBILITY = 4
    PRESENCE = 5
    FIXED = 6
    ASSIGNMENT = 7
    REMAINING_TASKS = 8
    INFEASIBLE = 9


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class Trail(EzPickle):
    """Undo log for ScheduleState."""

    active: bool

    marks: list[int]
    dep_marks: list[int]

    fields: list[int]
    tasks: list[TaskID]
    machines: list[MachineID]
    values: list[int]

    dep_log: list[tuple[TaskID, str, bool]]

    def __init__(self) -> None:
        self.active = False

        self.marks = []
        self.dep_marks = []

        self.fields = []
        self.tasks = []
        self.machines = []
        self.values = []

        self.dep_log = []

    @property
    def n_marks(self) -> int:
        """Return the number of stored checkpoint marks."""
        return len(self.marks)

    def clear(self) -> None:
        """Clear all the checkpoints and changes."""
        self.active = False

        self.marks.clear()
        self.dep_marks.clear()
        self.fields.clear()
        self.tasks.clear()
        self.machines.clear()
        self.values.clear()
        self.dep_log.clear()

    def mark(self) -> int:
        """Push a checkpoint. Activates recording on first call."""
        self.active = True
        mark = len(self.marks)

        self.marks.append(len(self.fields))
        self.dep_marks.append(len(self.dep_log))

        return mark

    def record(
        self,
        field: TrailField,
        old_value: int,
        task_id: TaskID = UNKNOWN_TASK,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        """Log an old value for later undo. No-op when inactive."""
        if not self.active:
            return

        self.fields.append(field.value)
        self.tasks.append(task_id)
        self.machines.append(machine_id)
        self.values.append(old_value)

    def record_dependency(
        self, task_id: TaskID, name: str, added: bool
    ) -> None:
        """Log a dependency-set mutation. `added=True` undoes by discarding `name`."""
        if not self.active:
            return

        self.dep_log.append((task_id, name, added))

    def has_mark(self, mark: int) -> bool:
        """Return whether a checkpoint mark exists."""
        return -len(self.marks) <= mark < len(self.marks)

    def __eq__(self, value: object, /) -> bool:
        """Return equality between trails."""
        return (
            isinstance(value, Trail)
            and self.active == value.active
            and self.marks == value.marks
            and self.dep_marks == value.dep_marks
            and self.fields == value.fields
            and self.tasks == value.tasks
            and self.machines == value.machines
            and self.values == value.values
            and self.dep_log == value.dep_log
        )

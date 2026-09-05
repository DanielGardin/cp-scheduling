"""CSP domain containers for scheduling variables."""

from enum import Enum

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    MIN_TIME,
    EzPickle,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.utils import flatten_matrix


class Presence(Enum):
    """Domain values for optional task presence."""

    INFEASIBLE = 0b00
    "Task cannot be present nor absent, domain wipeout (infeasible)."

    PRESENT = 0b01
    "Task must be present in the final schedule."

    ABSENT = 0b10
    "Task must be absent from the final schedule."

    UNDEFINED = 0b11
    "Task may be present or absent (initial value for optional tasks)."

    def contains_present(self) -> bool:
        """Return whether its presence can be PRESENT."""
        return bool(self.value & 0b01)

    def contains_absent(self) -> bool:
        """Return whether its presence can be PRESENT."""
        return bool(self.value & 0b10)


INFEASIBLE = Presence.INFEASIBLE
PRESENT = Presence.PRESENT
ABSENT = Presence.ABSENT
UNDEFINED = Presence.UNDEFINED


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False, acyclic=True)
class Bounds(EzPickle):
    """Integer bound container used for start/end variables."""

    pad: int

    lbs: list[Time]
    global_lbs: list[Time]

    ubs: list[Time]
    global_ubs: list[Time]

    def __init__(self, n_tasks: int, n_machines: int) -> None:
        """Initialize the Bounds container with a problem instance.

        Parameters
        ----------
        n_tasks: int
            The number of tasks in the problem instance.

        n_machines: int
            The number of machines in the problem instance.

        """
        nm = n_tasks * n_machines

        lbs = [MAX_TIME] * nm
        global_lbs = [MAX_TIME] * n_tasks

        ubs = [MIN_TIME] * nm
        global_ubs = [MIN_TIME] * n_tasks

        self.pad = n_machines
        self.lbs = lbs
        self.global_lbs = global_lbs
        self.ubs = ubs
        self.global_ubs = global_ubs

    def get_global_lb(self, task_id: TaskID) -> Time:
        """Get the global lower bound for a task."""
        return self.global_lbs[task_id]

    def get_global_ub(self, task_id: TaskID) -> Time:
        """Get the global upper bound for a task."""
        return self.global_ubs[task_id]

    def get_lb(self, task_id: TaskID, machine_id: MachineID) -> Time:
        """Get the lower bound for a task on a specific machine."""
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_lbs[task_id]

        return self.lbs[task_id * self.pad + machine_id]

    def get_ub(self, task_id: TaskID, machine_id: MachineID) -> Time:
        """Get the upper bound for a task on a specific machine."""
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_ubs[task_id]

        return self.ubs[task_id * self.pad + machine_id]

    def __eq__(self, value: object, /) -> bool:
        """Check equality of Bounds containers."""
        return (
            isinstance(value, Bounds)
            and self.pad == value.pad
            and self.lbs == value.lbs
            and self.global_lbs == value.global_lbs
            and self.ubs == value.ubs
            and self.global_ubs == value.global_ubs
        )


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False, acyclic=True)
class SparseFeasibleSet(EzPickle):
    """Reversible sparse-set domain over machine ids, per task."""

    pad: int

    order: list[MachineID]
    sparse: list[int]

    offsets: list[int]
    sizes: list[int]

    def __init__(self, machine_mask: list[list[bool]]) -> None:
        n_tasks = len(machine_mask)
        n_machines = len(machine_mask[0]) if n_tasks else 0
        self.pad = n_machines

        offsets = [0] * (n_tasks + 1)
        sizes = [0] * n_tasks
        for task_id in range(n_tasks):
            n_feasible = sum(machine_mask[task_id])
            sizes[task_id] = n_feasible
            offsets[task_id + 1] = offsets[task_id] + n_feasible

        order: list[MachineID] = [0] * offsets[n_tasks]
        sparse = [0] * (n_tasks * n_machines)

        for task_id in range(n_tasks):
            pos = offsets[task_id]
            row = task_id * n_machines

            for machine_id, feasible in enumerate(machine_mask[task_id]):
                if feasible:
                    order[pos] = machine_id
                    sparse[row + machine_id] = pos
                    pos += 1

        self.order = order
        self.sparse = sparse
        self.offsets = offsets
        self.sizes = sizes

    def size(self, task_id: TaskID) -> int:
        """Return the number of feasible machines for a task."""
        return self.sizes[task_id]

    def is_feasible(self, task_id: TaskID, machine_id: MachineID) -> bool:
        """Return whether machine_id is currently feasible for task_id. O(1)."""
        start = self.offsets[task_id]
        end = start + self.sizes[task_id]
        pos = self.sparse[task_id * self.pad + machine_id]

        return start <= pos < end and self.order[pos] == machine_id

    def forbid(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Remove machine_id if present. Returns the old size, or None. O(1)."""
        row = task_id * self.pad
        size = self.sizes[task_id]
        end = self.offsets[task_id] + size
        last = end - 1

        pos = self.sparse[row + machine_id]
        last_machine = self.order[last]

        order = self.order
        order[pos] = last_machine
        order[last] = machine_id

        self.sparse[row + last_machine] = pos

        self.sizes[task_id] = size - 1

    def restore_size(self, task_id: TaskID, old_size: int) -> None:
        """Undo: reset the boundary. O(1), no data movement."""
        self.sizes[task_id] = old_size

    def bounds(self, task_id: TaskID) -> tuple[int, int]:
        """Return (start, end) into `order` for this task's live machines."""
        start = self.offsets[task_id]
        return start, start + self.sizes[task_id]

    def __eq__(self, value: object, /) -> bool:
        """Check equality of SparseFeasibleSet containers."""
        return (
            isinstance(value, SparseFeasibleSet)
            and self.pad == value.pad
            and self.offsets == value.offsets
            and self.sizes == value.sizes
            and self.order == value.order
        )


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False, acyclic=True)
class TaskDomains(EzPickle):
    """Aggregate container for task variables used by the CSP kernel."""

    pad: int

    machines: SparseFeasibleSet
    remaining_times: list[Time]

    assignment: list[MachineID]
    presence: list[Presence]

    start: Bounds
    end: Bounds

    dependencies: list[set[str]]

    fixed: list[bool]

    def __init__(self, instance: ProblemInstance) -> None:
        n_tasks = instance.n_tasks
        n_machines = instance.n_machines
        self.pad = n_machines

        self.machines = SparseFeasibleSet(instance.machine_mask)

        remaining_times = flatten_matrix(instance.processing_times)
        self.remaining_times = remaining_times

        self.assignment = [GLOBAL_MACHINE_ID] * n_tasks
        self.presence = [
            UNDEFINED if optional else PRESENT for optional in instance.optional
        ]

        start = Bounds(n_tasks, n_machines)
        end = Bounds(n_tasks, n_machines)

        self.start = start
        self.end = end

        self.dependencies = [set() for _ in range(n_tasks)]
        self.fixed = [False] * n_tasks

        order = self.machines.order
        for task_id in range(n_tasks):
            start_idx, end_idx = self.machines.bounds(task_id)

            for i in range(start_idx, end_idx):
                machine_id = order[i]
                idx = task_id * n_machines + machine_id
                p = remaining_times[idx]

                start.lbs[idx] = MIN_TIME
                start.ubs[idx] = MAX_TIME - p
                end.lbs[idx] = MIN_TIME + p
                end.ubs[idx] = MAX_TIME

        for task_id in range(n_tasks):
            start.global_lbs[task_id] = MIN_TIME
            self.recompute_global_start_ubs(task_id)
            self.recompute_global_end_lbs(task_id)
            end.global_ubs[task_id] = MAX_TIME

    def recompute_global_start_lbs(self, task_id: TaskID) -> None:
        """Recompute the global lower bound for the start variable of a task."""
        lbs = self.start.lbs
        order = self.machines.order
        start_idx, end_idx = self.machines.bounds(task_id)
        row = task_id * self.pad

        global_lb = MAX_TIME
        for i in range(start_idx, end_idx):
            idx = row + order[i]
            lb = lbs[idx]
            if lb < global_lb:
                global_lb = lb

        self.start.global_lbs[task_id] = global_lb

    def recompute_global_start_ubs(self, task_id: TaskID) -> None:
        """Recompute the global upper bound for the start variable of a task."""
        ubs = self.start.ubs
        order = self.machines.order
        start_idx, end_idx = self.machines.bounds(task_id)
        row = task_id * self.pad

        global_ub = MIN_TIME
        for i in range(start_idx, end_idx):
            idx = row + order[i]
            ub = ubs[idx]
            if ub > global_ub:
                global_ub = ub

        self.start.global_ubs[task_id] = global_ub

    def recompute_global_end_lbs(self, task_id: TaskID) -> None:
        """Recompute the global lower bound for the end variable of a task."""
        lbs = self.end.lbs
        order = self.machines.order
        start_idx, end_idx = self.machines.bounds(task_id)
        row = task_id * self.pad

        global_lb = MAX_TIME
        for i in range(start_idx, end_idx):
            idx = row + order[i]
            lb = lbs[idx]
            if lb < global_lb:
                global_lb = lb

        self.end.global_lbs[task_id] = global_lb

    def recompute_global_end_ubs(self, task_id: TaskID) -> None:
        """Recompute the global upper bound for the end variable of a task."""
        ubs = self.end.ubs
        order = self.machines.order
        start_idx, end_idx = self.machines.bounds(task_id)
        row = task_id * self.pad

        global_ub = MIN_TIME
        for i in range(start_idx, end_idx):
            idx = row + order[i]
            ub = ubs[idx]
            if ub > global_ub:
                global_ub = ub

        self.end.global_ubs[task_id] = global_ub

    def recompute_all_global_bounds(self, task_id: TaskID) -> None:
        """Recompute all four global bounds (start/end, lb/ub) for a task."""
        self.recompute_global_start_ubs(task_id)
        self.recompute_global_start_lbs(task_id)
        self.recompute_global_end_ubs(task_id)
        self.recompute_global_end_lbs(task_id)

    def restore_task(self, task_id: TaskID) -> None:
        """Recompute derived global bounds for a task after a rollback."""
        self.recompute_all_global_bounds(task_id)

    def __eq__(self, value: object, /) -> bool:
        """Check equality of TaskDomains containers."""
        return (
            isinstance(value, TaskDomains)
            and self.pad == value.pad
            and self.machines == value.machines
            and self.remaining_times == value.remaining_times
            and self.assignment == value.assignment
            and self.presence == value.presence
            and self.start == value.start
            and self.end == value.end
            and self.dependencies == value.dependencies
            and self.fixed == value.fixed
        )

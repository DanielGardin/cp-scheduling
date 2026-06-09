"""CSP domain containers for scheduling variables."""

from typing import Final, Literal

from mypy_extensions import mypyc_attr
from typing_extensions import assert_never

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    MIN_TIME,
    Enum,
    EzPickle,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.instance import ProblemInstance

PresenceType = Literal[0b00, 0b01, 0b10, 0b11]


class Presence(Enum):
    """Domain values for optional task presence.

    The presence domain is represented by a two bit integer XY,
    where:

    X = 1 if the task can be present in the schedule, 0 otherwise.
    Y = 1 if the task can be absent from the schedule, 0 otherwise.

    In this sense, the possible domain values are:
    - 00 = {}, infeasible domain, the task cannot be present nor absent.
    - 01 = {present}, the task has to be present in the schedule to satisfy the constraints.
    - 10 = {absent}, the task has to be absent from the schedule to satisfy the constraints
    - 11 = {present, absent}, the task can be either present or absent.

    """

    INFEASIBLE: Final[Literal[0b00]] = 0b00
    "Task cannot be present nor absent, domain wipeout (infeasible)."

    PRESENT: Final[Literal[0b01]] = 0b01
    "Task must be present in the final schedule."

    ABSENT: Final[Literal[0b10]] = 0b10
    "Task must be absent from the final schedule."

    UNDEFINED: Final[Literal[0b11]] = 0b11
    "Task may be present or absent (initial value for optional tasks)."


INFEASIBLE = Presence.INFEASIBLE
PRESENT = Presence.PRESENT
ABSENT = Presence.ABSENT
UNDEFINED = Presence.UNDEFINED


def presence_to_str(presence: PresenceType) -> str:
    """Return the string name for a presence flag."""
    if presence == INFEASIBLE:
        return "INFEASIBLE"

    if presence == PRESENT:
        return "PRESENT"

    if presence == ABSENT:
        return "ABSENT"

    if presence == UNDEFINED:
        return "UNDEFINED"

    assert_never(presence)


# FUTURE: Consider going back to store bounds in nested lists (list[list[Time]])
# The access pattern is almost always task_id -> machine_id, which is trivialized
# by tlb = lb[task_id], and then tlb[machine_id].
@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class Bounds(EzPickle):
    """Integer bound container used for start/end variables.

    The container stores per-(task,machine) lower/upper bounds as flat lists
    (row-major by task) as well as cached global bounds for fast queries.

    Notes
    -----
    Do not mutate these lists directly, use :class:`ScheduleState` APIs which
    ensure correctness and emit domain events.

    """

    n_machines: int

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

        lbs = [MAX_TIME] * (nm)
        global_lbs = [MAX_TIME] * n_tasks

        ubs = [MIN_TIME] * (nm)
        global_ubs = [MIN_TIME] * n_tasks

        self.n_machines = n_machines
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

        return self.lbs[task_id * self.n_machines + machine_id]

    def get_ub(self, task_id: TaskID, machine_id: MachineID) -> Time:
        """Get the upper bound for a task on a specific machine."""
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_ubs[task_id]

        return self.ubs[task_id * self.n_machines + machine_id]

    def __eq__(self, value: object, /) -> bool:
        """Check equality of Bounds containers."""
        return (
            isinstance(value, Bounds)
            and self.n_machines == value.n_machines
            and self.lbs == value.lbs
            and self.global_lbs == value.global_lbs
            and self.ubs == value.ubs
            and self.global_ubs == value.global_ubs
        )


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class TaskDomains(EzPickle):
    """Aggregate container for task variables used by the CSP kernel."""

    feasible_machines: list[set[MachineID]]
    remaining_times: list[Time]

    assignment: list[MachineID]
    presence: list[PresenceType]

    start: Bounds
    end: Bounds

    def __init__(self, instance: ProblemInstance) -> None:
        """Initialize the TaskDomains with a problem instance.

        Parameters
        ----------
        instance: ProblemInstance
            The problem instance containing tasks, machines, processing times, etc.

        """
        n_tasks = instance.n_tasks
        n_machines = instance.n_machines

        feasible_machines = [
            {
                machine_id
                for machine_id, is_feasible in enumerate(machine_mask)
                if is_feasible
            }
            for machine_mask in instance.machine_mask
        ]
        remaining_times = [MAX_TIME] * (n_tasks * n_machines)

        assignment = [GLOBAL_MACHINE_ID] * n_tasks
        presence: list[PresenceType] = [
            UNDEFINED if optional else PRESENT for optional in instance.optional
        ]

        start = Bounds(n_tasks, n_machines)
        end = Bounds(n_tasks, n_machines)

        self.feasible_machines = feasible_machines
        self.remaining_times = remaining_times

        self.assignment = assignment
        self.presence = presence

        self.start = start
        self.end = end

        processing_times = instance.processing_times

        for task_id in range(n_tasks):
            p_times = processing_times[task_id]

            start_idx = task_id * n_machines
            for machine_id in feasible_machines[task_id]:
                idx = start_idx + machine_id

                p = p_times[machine_id]

                start.lbs[idx] = MIN_TIME
                start.ubs[idx] = MAX_TIME - p
                end.lbs[idx] = MIN_TIME + p
                end.ubs[idx] = MAX_TIME
                remaining_times[idx] = p

            start.global_lbs[task_id] = MIN_TIME
            self.recompute_global_start_ubs(task_id)
            self.recompute_global_end_lbs(task_id)
            end.global_ubs[task_id] = MAX_TIME

    def recompute_global_start_lbs(self, task_id: TaskID) -> None:
        """Recompute the global lower bound for the start variable of a task."""
        start = self.start
        row = task_id * start.n_machines

        global_lb = MAX_TIME
        for machine_id in self.feasible_machines[task_id]:
            lb = start.lbs[row + machine_id]
            if lb < global_lb:
                global_lb = lb

        start.global_lbs[task_id] = global_lb

    def recompute_global_start_ubs(self, task_id: TaskID) -> None:
        """Recompute the global upper bound for the start variable of a task."""
        start = self.start
        row = task_id * start.n_machines

        global_ub = MIN_TIME
        for machine_id in self.feasible_machines[task_id]:
            ub = start.ubs[row + machine_id]
            if ub > global_ub:
                global_ub = ub

        start.global_ubs[task_id] = global_ub

    def recompute_global_end_lbs(self, task_id: TaskID) -> None:
        """Recompute the global lower bound for the end variable of a task."""
        end = self.end
        row = task_id * end.n_machines

        global_lb = MAX_TIME
        for machine_id in self.feasible_machines[task_id]:
            lb = end.lbs[row + machine_id]
            if lb < global_lb:
                global_lb = lb

        end.global_lbs[task_id] = global_lb

    def recompute_global_end_ubs(self, task_id: TaskID) -> None:
        """Recompute the global upper bound for the end variable of a task."""
        end = self.end
        row = task_id * end.n_machines

        global_ub = MIN_TIME
        for machine_id in self.feasible_machines[task_id]:
            ub = end.ubs[row + machine_id]
            if ub > global_ub:
                global_ub = ub

        end.global_ubs[task_id] = global_ub

    def is_machine_feasible(
        self, task_id: TaskID, machine_id: MachineID
    ) -> bool:
        """Check if a machine is currently feasible for a task."""
        return machine_id in self.feasible_machines[task_id]

    def get_feasible_machines(self, task_id: TaskID) -> tuple[MachineID, ...]:
        """Return the tuple of currently feasible machines for a task."""
        return tuple(self.feasible_machines[task_id])

    def __eq__(self, value: object, /) -> bool:
        """Check equality of TaskDomains containers."""
        return (
            isinstance(value, TaskDomains)
            and self.feasible_machines == value.feasible_machines
            and self.remaining_times == value.remaining_times
            and self.assignment == value.assignment
            and self.presence == value.presence
            and self.start == value.start
            and self.end == value.end
        )

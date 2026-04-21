from typing import Literal, Final
from typing_extensions import assert_never

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    MachineID, TaskID, Time,
    GLOBAL_MACHINE_ID, MAX_TIME, MIN_TIME,
    Enum, EzPickle,
)

from cpscheduler.environment.state.instance import ProblemInstance

DUMMY_INSTANCE = ProblemInstance({})

PresenceType = Literal[0b00, 0b01, 0b10, 0b11]

class Presence(Enum):
    """
    Presence is a domain for optional tasks, when they can be either present or
    absent from the schedule.

    It can be represented as a two bit integer
    XY
    where:
    X = 1 if the task can be present in the schedule, 0 otherwise
    Y = 1 if the task can be absent from the schedule, 0 otherwise

    In this sense, the possible domain values are:
    00 = {}, infeasible domain, the task cannot be present nor absent.
    01 = {present}, the task has to be present in the schedule to satisfy the constraints.
    10 = {absent}, the task has to be absent from the schedule to satisfy the constraints.
    11 = {present, absent}, the task can be either present or absent.
    """

    INFEASIBLE: Final[Literal[0b00]] = 0b00
    "Task is not consistent with the constraints and cannot be scheduled."

    PRESENT: Final[Literal[0b01]] = 0b01
    "Task has to be present in the schedule to satisfy the constraints."

    ABSENT: Final[Literal[0b10]] = 0b10
    "Task has to be absent from the schedule to satisfy the constraints."

    UNDEFINED: Final[Literal[0b11]] = 0b11
    "Task presence has not been determined yet. Initial value for optional tasks."


INFEASIBLE = Presence.INFEASIBLE
PRESENT = Presence.PRESENT
ABSENT = Presence.ABSENT
UNDEFINED = Presence.UNDEFINED


def presence_to_str(presence: PresenceType) -> str:
    if presence == INFEASIBLE:
        return "INFEASIBLE"

    elif presence == PRESENT:
        return "PRESENT"

    elif presence == ABSENT:
        return "ABSENT"

    elif presence == UNDEFINED:
        return "UNDEFINED"

    assert_never(presence)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class Bounds(EzPickle):
    """
    Container for integer bounds in the scheduling environment.

    Each variable (e.g., start time, end time) in a Constraint Programming model
    has a domain set of values that are consistent with the constraints of the
    problem.

    The Bounds class maintains interval domains for each variable, managing both
    lower and upper bounds for each task-machine pair, as well as global bounds
    for each task, defined as

    - global_lb(task) = min(lb(task, machine) for machine in machines)
    - global_ub(task) = max(ub(task, machine) for machine in machines)

    ## IMPORTANT
    Never acess or modify the bounds directly outside of ScheduleState to ensure
    consistency.
    """

    n_machines: int

    lbs: list[Time]
    global_lbs: list[Time]

    ubs: list[Time]
    global_ubs: list[Time]

    def __init__(self, instance: ProblemInstance | None = None) -> None:
        if instance is None:
            instance = DUMMY_INSTANCE

        n_tasks = instance.n_tasks
        n_machines = instance.n_machines

        lbs = [MAX_TIME] * (n_tasks * n_machines)
        global_lbs = [MAX_TIME] * n_tasks

        ubs = [MIN_TIME] * (n_tasks * n_machines)
        global_ubs = [MIN_TIME] * n_tasks

        for task_id, p_times in enumerate(instance.processing_times):
            if not p_times:
                continue

            for machine_id in p_times:
                lbs[task_id * n_machines + machine_id] = MIN_TIME
                ubs[task_id * n_machines + machine_id] = MAX_TIME

            global_lbs[task_id] = MIN_TIME
            global_ubs[task_id] = MAX_TIME

        self.n_machines = n_machines
        self.lbs = lbs
        self.global_lbs = global_lbs
        self.ubs = ubs
        self.global_ubs = global_ubs

    def get_global_lb(self, task_id: TaskID) -> Time:
        return self.global_lbs[task_id]

    def get_global_ub(self, task_id: TaskID) -> Time:
        return self.global_ubs[task_id]

    def get_lb(self, task_id: TaskID, machine_id: MachineID) -> Time:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_lbs[task_id]

        return self.lbs[task_id * self.n_machines + machine_id]

    def get_ub(self, task_id: TaskID, machine_id: MachineID) -> Time:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_ubs[task_id]

        return self.ubs[task_id * self.n_machines + machine_id]

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, Bounds):
            return False
        
        return (
            self.n_machines ==value.n_machines
            and self.lbs ==value.lbs
            and self.global_lbs ==value.global_lbs
            and self.ubs ==value.ubs
            and self.global_ubs ==value.global_ubs
        )


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class TaskDomains(EzPickle):
    """
    Container for the task variables in the scheduling environment.

    Do not modify these variables directly, use the appropriate methods in
    ScheduleState to ensure consistency and proper updates of the bounds and
    feasibility checks.
    """

    original_machines: list[tuple[MachineID, ...]]

    feasible_machines: list[set[MachineID]]
    remaining_times: list[Time]

    assignment: list[MachineID]
    presence: list[PresenceType]

    start: Bounds
    end: Bounds

    def __init__(self, instance: ProblemInstance | None = None) -> None:
        if instance is None:
            instance = DUMMY_INSTANCE

        n_tasks = instance.n_tasks
        n_machines = instance.n_machines

        remaining_times = [0] * (n_tasks * n_machines)

        presence: list[PresenceType] = [
            UNDEFINED if optional else PRESENT for optional in instance.optional
        ]

        feasible_machines: list[set[MachineID]] = [set() for _ in range(n_tasks)]
        start = Bounds(instance)
        end = Bounds(instance)


        for task_id, p_times in enumerate(instance.processing_times):
            machines = feasible_machines[task_id]

            start_idx = task_id * n_machines
            for machine, processing_time in p_times.items():
                machines.add(machine)

                idx = start_idx + machine

                remaining_times[idx] = processing_time
                start.ubs[idx] = end.ubs[idx] - processing_time
                end.lbs[idx] = start.lbs[idx] + processing_time

        self.original_machines = [
            tuple(p_times) for p_times in instance.processing_times
        ]

        self.feasible_machines = feasible_machines
        self.remaining_times = remaining_times

        self.assignment = [GLOBAL_MACHINE_ID] * n_tasks
        self.presence = presence

        self.start = start
        self.end = end


    def get_feasible_machines(self, task_id: TaskID) -> tuple[MachineID, ...]:
        return tuple(self.feasible_machines[task_id])

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, TaskDomains):
            return False
        
        return (
            self.original_machines == value.original_machines
            and self.feasible_machines == value.feasible_machines
            and self.remaining_times == value.remaining_times
            and self.assignment == value.assignment
            and self.presence == value.presence
            and self.start == value.start
            and self.end == value.end
        )

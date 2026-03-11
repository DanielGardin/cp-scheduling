from typing import Any, TypeAlias, Final

from mypy_extensions import u8

from cpscheduler.environment.constants import (
    MachineID,
    TaskID,
    Time,
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    MIN_TIME
)

from cpscheduler.environment.state.events import Event, VarField
from cpscheduler.environment.state.instance import ProblemInstance


DUMMY_INSTANCE = ProblemInstance({})

# Helper constants to avoid calling getattr on enums

ASSIGNMENT = VarField.ASSIGNMENT
START_LB = VarField.START_LB
START_UB = VarField.START_UB
END_LB = VarField.END_LB
END_UB = VarField.END_UB
PRESENCE = VarField.PRESENCE
ABSENCE = VarField.ABSENCE
INFEASIBLE = VarField.INFEASIBLE

PresenceType: TypeAlias = u8

class Presence:
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

    __slots__ = ()

    INFEASIBLE: Final[PresenceType] = 0b00
    "Task is not consistent with the constraints and cannot be scheduled."

    PRESENT: Final[PresenceType] = 0b01
    "Task has to be present in the schedule to satisfy the constraints."

    ABSENT: Final[PresenceType] = 0b10
    "Task has to be absent from the schedule to satisfy the constraints."

    UNDEFINED: Final[PresenceType] = 0b11
    "Task presence has not been determined yet. Initial value for optional tasks."

PRESENCE_INFEASIBLE = Presence.INFEASIBLE
PRESENT = Presence.PRESENT
ABSENT = Presence.ABSENT
UNDEFINED = Presence.UNDEFINED

class Bounds:
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

    __slots__ = (
        "n_machines",
        "lbs",
        "global_lbs",
        "ubs",
        "global_ubs",
    )

    n_machines: int

    lbs: list[Time]
    global_lbs: list[Time]

    ubs: list[Time]
    global_ubs: list[Time]

    def __init__(self, instance: ProblemInstance) -> None:
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

    def get_lb(self, task_id: TaskID, machine_id: MachineID) -> Time:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_lbs[task_id]

        return self.lbs[task_id * self.n_machines + machine_id]

    def get_ub(self, task_id: TaskID, machine_id: MachineID) -> Time:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_ubs[task_id]

        return self.ubs[task_id * self.n_machines + machine_id]

    def recompute_global_lb(self, task_id: TaskID, feasible_machines: list[MachineID]) -> None:
        row = task_id * self.n_machines

        min_value = MAX_TIME
        for m_id in feasible_machines:
            idx = row + m_id

            if self.lbs[idx] < min_value:
                min_value = self.lbs[idx]

        self.global_lbs[task_id] = min_value

    def recompute_global_ub(self, task_id: TaskID, feasible_machines: list[MachineID]) -> None:
        row = task_id * self.n_machines

        max_value = MIN_TIME
        for m_id in feasible_machines:
            idx = row + m_id

            if self.ubs[idx] > max_value:
                max_value = self.ubs[idx]

        self.global_ubs[task_id] = max_value

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Bounds):
            return NotImplemented

        return (
            self.n_machines == value.n_machines and
            self.lbs == value.lbs and
            self.global_lbs == value.global_lbs and
            self.ubs == value.ubs and
            self.global_ubs == value.global_ubs
        )

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.n_machines,
            self.lbs,
            self.global_lbs,
            self.ubs,
            self.global_ubs,
        )
        return (self.__class__, (DUMMY_INSTANCE,), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.n_machines,
            self.lbs,
            self.global_lbs,
            self.ubs,
            self.global_ubs,
        ) = state

class ScheduleVariables:
    """
    Container for the task variables in the scheduling environment.

    Do not modify these variables directly, use the appropriate methods in
    ScheduleState to ensure consistency and proper updates of the bounds and
    feasibility checks.
    """

    __slots__ = [
        "remaining_times",
        "feasible_machines",
        "assignment",
        "presence",
        "feasible",
        "fixed",
        "start",
        "end",
    ]

    remaining_times: list[Time]
    feasible_machines: list[list[MachineID]]

    assignment: list[MachineID]
    presence: list[u8]

    start: Bounds
    end: Bounds

    feasible: list[bool]
    fixed: list[bool]

    def __init__(self, instance: ProblemInstance) -> None:
        n_tasks = instance.n_tasks
        n_machines = instance.n_machines

        remaining_times = [MAX_TIME] * (n_tasks * n_machines)

        presence = [
            UNDEFINED if optional else PRESENT
            for optional in instance.optional
        ]

        feasible_machines: list[list[MachineID]] = [[] for _ in range(n_tasks)]

        start = Bounds(instance)
        end = Bounds(instance)

        for task_id, p_times in enumerate(instance.processing_times):
            feasible_machines[task_id] = list(p_times.keys())

            start_idx = task_id * n_machines
            for machine, processing_time in p_times.items():
                idx = start_idx + machine

                remaining_times[idx] = processing_time
                start.ubs[idx] = end.ubs[idx] - processing_time
                end.lbs[idx] = start.lbs[idx] + processing_time

            start.recompute_global_ub(task_id, feasible_machines[task_id])
            end.recompute_global_lb(task_id, feasible_machines[task_id])

        self.remaining_times = remaining_times
        self.feasible_machines = feasible_machines
        self.presence = presence
        self.start = start
        self.end = end

        self.assignment = [GLOBAL_MACHINE_ID] * n_tasks
        self.feasible = [True] * n_tasks
        self.fixed = [False] * n_tasks

    def restrict_presence(self, task_id: TaskID, mask: PresenceType) -> Event | None:
        old_presence = self.presence[task_id]
        new_presence = old_presence & mask

        if new_presence == old_presence:
            return None

        self.presence[task_id] = new_presence
        if new_presence == PRESENCE_INFEASIBLE:
            self.feasible[task_id] = False
            return Event(task_id, INFEASIBLE)

        return Event(
            task_id,
            PRESENCE if new_presence == PRESENT else ABSENCE,
        )

    def restrict_machine(self, task_id: TaskID, machine_id: MachineID) -> Event | None:
        if machine_id in self.feasible_machines[task_id]:
            self.set_start_lb(task_id, MAX_TIME, machine_id)
            return Event(task_id, INFEASIBLE, machine_id)

        return None

    def set_start_lb(
        self,
        task_id: TaskID,
        lb: Time,
        machine_id: MachineID,
    ) -> Event | None:
        if lb <= self.start.get_lb(task_id, machine_id):
            return None

        row = task_id * self.start.n_machines
        feasible_machines = self.feasible_machines[task_id]

        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_lbs = self.end.lbs
        end_ubs = self.end.ubs
        remaining_times = self.remaining_times

        start_global_lbs = self.start.global_lbs

        if machine_id != GLOBAL_MACHINE_ID:
            idx = row + machine_id
    
            old_lb = start_lbs[idx]
            end_lb = lb + remaining_times[idx]

            start_lbs[idx] = lb
            end_lbs[idx] = end_lb

            if lb > start_ubs[idx] or end_lb > end_ubs[idx]:
                feasible_machines.remove(machine_id)

                self.start.recompute_global_lb(task_id, feasible_machines)
                self.end.recompute_global_lb(task_id, feasible_machines)

                if not feasible_machines:
                    self.feasible[task_id] = False
                    return Event(task_id, INFEASIBLE)

                return Event(task_id, INFEASIBLE, machine_id)

            if old_lb == start_global_lbs[task_id]:
                self.start.recompute_global_lb(task_id, feasible_machines)
                self.end.recompute_global_lb(task_id, feasible_machines)
            
            return Event(task_id, START_LB, machine_id)

        for m_id in feasible_machines[:]:
            idx = row + m_id

            if start_lbs[idx] < lb:
                start_lbs[idx] = lb
                end_lbs[idx] = lb + remaining_times[idx]

                if lb > start_ubs[idx] or end_lbs[idx] > end_ubs[idx]:
                    feasible_machines.remove(m_id)

        self.start.recompute_global_lb(task_id, feasible_machines)
        self.end.recompute_global_lb(task_id, feasible_machines)

        if not feasible_machines:
            self.feasible[task_id] = False
            return Event(task_id, INFEASIBLE)

        return Event(task_id, START_LB)

    def set_start_ub(
        self,
        task_id: TaskID,
        ub: Time,
        machine_id: MachineID,
    ) -> Event | None:
        if ub >= self.start.get_ub(task_id, machine_id):
            return None

        row = task_id * self.start.n_machines
        feasible_machines = self.feasible_machines[task_id]

        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_lbs = self.end.lbs
        end_ubs = self.end.ubs
        remaining_times = self.remaining_times

        if machine_id != GLOBAL_MACHINE_ID:
            idx = row + machine_id

            start_lb = start_lbs[idx]
            end_ub = ub + remaining_times[idx]

            if ub < start_lb or end_ub < end_lbs[idx]:
                feasible_machines.remove(machine_id)

                self.start.recompute_global_ub(task_id, feasible_machines)
                self.end.recompute_global_ub(task_id, feasible_machines)

                if not feasible_machines:
                    self.feasible[task_id] = False
                    return Event(task_id, INFEASIBLE)

                return Event(task_id, INFEASIBLE, machine_id)

            old_ub = start_ubs[idx]

            start_ubs[idx] = ub
            end_ubs[idx] = end_ub

            if old_ub == self.start.global_ubs[task_id]:
                self.start.recompute_global_ub(task_id, feasible_machines)
                self.end.recompute_global_ub(task_id, feasible_machines)

            return Event(task_id, START_UB, machine_id)

        for m_id in feasible_machines[:]:
            idx = row + m_id

            if start_ubs[idx] > ub:
                new_end_ub = ub + remaining_times[idx]

                if ub < start_lbs[idx] or new_end_ub < end_lbs[idx]:
                    feasible_machines.remove(m_id)
                else:
                    start_ubs[idx] = ub
                    end_ubs[idx] = new_end_ub

        self.start.recompute_global_ub(task_id, feasible_machines)
        self.end.recompute_global_ub(task_id, feasible_machines)

        if not feasible_machines:
            self.feasible[task_id] = False
            return Event(task_id, INFEASIBLE)

        return Event(task_id, START_UB)

    def set_end_lb(
        self,
        task_id: TaskID,
        lb: Time,
        machine_id: MachineID,
    ) -> Event | None:
        if lb <= self.end.get_lb(task_id, machine_id):
            return None

        row = task_id * self.end.n_machines
        feasible_machines = self.feasible_machines[task_id]

        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_lbs = self.end.lbs
        end_ubs = self.end.ubs
        remaining_times = self.remaining_times

        if machine_id != GLOBAL_MACHINE_ID:
            idx = row + machine_id

            start_lb = lb - remaining_times[idx]

            if lb > end_ubs[idx] or start_lb > start_ubs[idx]:
                feasible_machines.remove(machine_id)

                self.start.recompute_global_lb(task_id, feasible_machines)
                self.end.recompute_global_lb(task_id, feasible_machines)

                if not feasible_machines:
                    self.feasible[task_id] = False
                    return Event(task_id, INFEASIBLE)

                return Event(task_id, INFEASIBLE, machine_id)

            old_lb = end_lbs[idx]

            end_lbs[idx] = lb
            start_lbs[idx] = start_lb

            if old_lb == self.end.global_lbs[task_id]:
                self.start.recompute_global_lb(task_id, feasible_machines)
                self.end.recompute_global_lb(task_id, feasible_machines)

            return Event(task_id, END_LB, machine_id)

        for m_id in feasible_machines[:]:
            idx = row + m_id

            if end_lbs[idx] < lb:
                new_start_lb = lb - remaining_times[idx]

                if lb > end_ubs[idx] or new_start_lb > start_ubs[idx]:
                    feasible_machines.remove(m_id)

                else:
                    end_lbs[idx] = lb
                    start_lbs[idx] = new_start_lb

        self.start.recompute_global_lb(task_id, feasible_machines)
        self.end.recompute_global_lb(task_id, feasible_machines)

        if not feasible_machines:
            self.feasible[task_id] = False
            return Event(task_id, INFEASIBLE)

        return Event(task_id, END_LB)

    def set_end_ub(
        self,
        task_id: TaskID,
        ub: Time,
        machine_id: MachineID,
    ) -> Event | None:
        if ub >= self.end.get_ub(task_id, machine_id):
            return None

        row = task_id * self.end.n_machines
        feasible_machines = self.feasible_machines[task_id]

        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_lbs = self.end.lbs
        end_ubs = self.end.ubs
        remaining_times = self.remaining_times

        if machine_id != GLOBAL_MACHINE_ID:
            idx = row + machine_id

            start_ub = ub - remaining_times[idx]

            if ub < end_lbs[idx] or start_ub < start_lbs[idx]:
                feasible_machines.remove(machine_id)

                self.start.recompute_global_ub(task_id, feasible_machines)
                self.end.recompute_global_ub(task_id, feasible_machines)

                if not feasible_machines:
                    self.feasible[task_id] = False
                    return Event(task_id, INFEASIBLE)

                return Event(task_id, INFEASIBLE, machine_id)

            old_ub = end_ubs[idx]

            end_ubs[idx] = ub
            start_ubs[idx] = start_ub

            if old_ub == self.end.global_ubs[task_id]:
                self.start.recompute_global_ub(task_id, feasible_machines)
                self.end.recompute_global_ub(task_id, feasible_machines)

            return Event(task_id, END_UB, machine_id)

        for m_id in feasible_machines[:]:
            idx = row + m_id

            if end_ubs[idx] > ub:
                new_start_ub = ub - remaining_times[idx]

                if ub < end_lbs[idx] or new_start_ub < start_lbs[idx]:
                    feasible_machines.remove(m_id)

                else:
                    end_ubs[idx] = ub
                    start_ubs[idx] = new_start_ub

        self.start.recompute_global_ub(task_id, feasible_machines)
        self.end.recompute_global_ub(task_id, feasible_machines)

        if not feasible_machines:
            self.feasible[task_id] = False
            return Event(task_id, INFEASIBLE)

        return Event(task_id, END_UB)

    def assign(self, task_id: TaskID, time: Time, machine_id: MachineID) -> Event:
        row = task_id * self.start.n_machines
        idx = row + machine_id
        duration = self.remaining_times[idx]

        start_lb = self.start.lbs[idx]
        start_ub = self.start.ubs[idx]
        end_lb = self.end.lbs[idx]
        end_ub = self.end.ubs[idx]

        end_time = time + duration

        if (
            time < start_lb or
            time > start_ub or
            end_time < end_lb or
            end_time > end_ub
        ):
            self.feasible[task_id] = False
            return Event(task_id, INFEASIBLE)

        self.assignment[task_id] = machine_id
        self.fixed[task_id] = True

        self.start.lbs[idx] = time
        self.start.ubs[idx] = time
        self.end.lbs[idx] = end_time
        self.end.ubs[idx] = end_time

        self.start.global_lbs[task_id] = time
        self.start.global_ubs[task_id] = time
        self.end.global_lbs[task_id] = end_time
        self.end.global_ubs[task_id] = end_time

        self.feasible_machines[task_id].clear()
        self.feasible_machines[task_id].append(machine_id)

        return Event(task_id, ASSIGNMENT, machine_id)

    def __repr__(self) -> str:
        return (
            f"ScheduleVariables(remaining_times={self.remaining_times}, "
            f"assignment={self.assignment}, presence={self.presence}, "
            f"start={self.start}, end={self.end})"
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ScheduleVariables):
            return NotImplemented

        return (
            self.remaining_times == value.remaining_times and
            self.assignment == value.assignment and
            self.presence == value.presence and
            self.start == value.start and
            self.end == value.end
        )

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.remaining_times,
            self.assignment,
            self.presence,
            self.start,
            self.end,
        )
        return (self.__class__, (DUMMY_INSTANCE,), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.remaining_times,
            self.assignment,
            self.presence,
            self.start,
            self.end,
        ) = state

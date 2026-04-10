from typing import Any, Literal, Final

from cpscheduler.environment.constants import (
    MachineID,
    TaskID,
    Time,
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    MIN_TIME,
)

from cpscheduler.environment.state.events import (
    DomainEvent,
    VarField,
    VarFieldType,
)
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
MACHINE_INFEASIBLE = VarField.MACHINE_INFEASIBLE
PAUSE = VarField.PAUSE
BOUNDS_RESET = VarField.BOUNDS_RESET
STATE_INFEASIBLE = VarField.STATE_INFEASIBLE

PresenceType = Literal[0b00, 0b01, 0b10, 0b11]


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

    raise ValueError(f"Invalid presence value: {presence}.")


def can_be_present(presence: PresenceType) -> bool:
    return (presence & PRESENT) != 0


def can_be_absent(presence: PresenceType) -> bool:
    return (presence & ABSENT) != 0


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

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Bounds):
            return NotImplemented

        return (
            self.n_machines == value.n_machines
            and self.lbs == value.lbs
            and self.global_lbs == value.global_lbs
            and self.ubs == value.ubs
            and self.global_ubs == value.global_ubs
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


IntervalEnd = bool
START: IntervalEnd = True
END: IntervalEnd = False

Bound = bool
LB: Bound = True
UB: Bound = False


class ScheduleVariables:
    """
    Container for the task variables in the scheduling environment.

    Do not modify these variables directly, use the appropriate methods in
    ScheduleState to ensure consistency and proper updates of the bounds and
    feasibility checks.
    """

    __slots__ = [
        "remaining_times",
        "original_machines",
        "feasible_machines",
        "assignment",
        "presence",
        "fixed",
        "start",
        "end",
        "infeasible",
        "event_queue",
        "awaiting_tasks"
    ]

    remaining_times: list[Time]
    feasible_machines: list[list[MachineID]]
    original_machines: list[tuple[MachineID, ...]]

    assignment: list[MachineID]
    presence: list[PresenceType]

    start: Bounds
    end: Bounds

    fixed: list[bool]
    infeasible: bool

    event_queue: list[DomainEvent]

    # Cached containers
    awaiting_tasks: set[TaskID]

    def __init__(self, instance: ProblemInstance) -> None:
        n_tasks = instance.n_tasks
        n_machines = instance.n_machines

        remaining_times = [0] * (n_tasks * n_machines)

        presence: list[PresenceType] = [
            UNDEFINED if optional else PRESENT for optional in instance.optional
        ]

        feasible_machines: list[list[MachineID]] = [[] for _ in range(n_tasks)]
        start = Bounds(instance)
        end = Bounds(instance)

        self.feasible_machines = feasible_machines
        self.start = start
        self.end = end

        for task_id, p_times in enumerate(instance.processing_times):
            feasible_machines[task_id] = list(p_times.keys())

            start_idx = task_id * n_machines
            for machine, processing_time in p_times.items():
                idx = start_idx + machine

                remaining_times[idx] = processing_time
                start.ubs[idx] = end.ubs[idx] - processing_time
                end.lbs[idx] = start.lbs[idx] + processing_time

            self._recompute_global_bound(task_id, START, LB)
            self._recompute_global_bound(task_id, END, LB)

        self.original_machines = [
            tuple(p_times) for p_times in instance.processing_times
        ]

        self.remaining_times = remaining_times
        self.presence = presence

        self.assignment = [GLOBAL_MACHINE_ID] * n_tasks
        self.fixed = [False] * n_tasks
        self.infeasible = False

        self.event_queue = []

        self.awaiting_tasks = set(range(n_tasks))

    def set_infeasible_state(self) -> None:
        self.infeasible = True

        # Task -1 indicates that the infeasibility cause cannot be determined
        self.event_queue.append(DomainEvent(-1, STATE_INFEASIBLE))

    def _recompute_global_bound(
        self, task_id: TaskID, var: IntervalEnd, bound: Bound
    ) -> None:
        variable = self.start if var else self.end
        row = task_id * variable.n_machines

        feasible_machines = self.feasible_machines[task_id]

        best = MAX_TIME if bound else MIN_TIME
        if bound:  # bound == LB:
            lbs = variable.lbs

            for m_id in feasible_machines:
                idx = row + m_id

                value = lbs[idx]
                if value < best:
                    best = value

            variable.global_lbs[task_id] = best

        else:  # bound == UB:
            ubs = variable.ubs

            for m_id in feasible_machines:
                idx = row + m_id

                value = ubs[idx]
                if value > best:
                    best = value

            variable.global_ubs[task_id] = best

    def _restrict_presence(self, task_id: TaskID, mask: Literal[0b01, 0b10]) -> None:
        old_presence = self.presence[task_id]
        new_presence = old_presence & mask

        field: VarFieldType
        if new_presence == old_presence:
            return

        if new_presence == PRESENT:
            field = PRESENCE

        elif new_presence == ABSENT:
            self.awaiting_tasks.remove(task_id)
            field = ABSENCE

        elif new_presence == INFEASIBLE:
            # The task cannot be present nor absent, it is infeasible based on
            # the current propagation.
            self.infeasible = True
            field = STATE_INFEASIBLE

        else:
            raise RuntimeError(
                f"Unreachable: unexpected presence value {new_presence!r}"
            )

        self.presence[task_id] = new_presence

        self.event_queue.append(DomainEvent(task_id, field))

    def require_task(self, task_id: TaskID) -> None:
        "Require a task to be present in the schedule."
        self._restrict_presence(task_id, PRESENT)

    def forbid_task(self, task_id: TaskID) -> None:
        "Forbid a task from being present in the schedule."
        self._restrict_presence(task_id, ABSENT)

    def restrict_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        feasible_machines = self.feasible_machines[task_id]

        if machine_id not in feasible_machines:
            return

        feasible_machines.remove(machine_id)

        if not feasible_machines:
            self._restrict_presence(task_id, ABSENT)
            return

        for var in (START, END):
            for bound in (LB, UB):
                self._recompute_global_bound(task_id, var, bound)

        self.event_queue.append(
            DomainEvent(task_id, MACHINE_INFEASIBLE, machine_id)
        )

    def set_machine_start_lb(
        self, task_id: TaskID, lb: Time, machine_id: MachineID
    ) -> None:
        idx = task_id * self.start.n_machines + machine_id
        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_lbs = self.end.lbs
        end_ubs = self.end.ubs

        old_lb = start_lbs[idx]
        end_lb = lb + self.remaining_times[idx]

        start_lbs[idx] = lb
        end_lbs[idx] = end_lb

        if lb > start_ubs[idx] or end_lb > end_ubs[idx]:
            self.restrict_machine(task_id, machine_id)
            return

        if old_lb == self.start.global_lbs[task_id]:
            self._recompute_global_bound(task_id, START, LB)
            self._recompute_global_bound(task_id, END, LB)

        self.event_queue.append(DomainEvent(task_id, START_LB, machine_id))

    def set_start_lb(self, task_id: TaskID, lb: Time) -> None:
        row = task_id * self.start.n_machines

        feasible_machines = self.feasible_machines[task_id]

        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_lbs = self.end.lbs
        end_ubs = self.end.ubs
        remaining_times = self.remaining_times

        for m_id in feasible_machines[:]:
            idx = row + m_id

            if start_lbs[idx] < lb:
                start_lbs[idx] = lb
                end_lbs[idx] = lb + remaining_times[idx]

                if lb > start_ubs[idx] or end_lbs[idx] > end_ubs[idx]:
                    self.restrict_machine(task_id, m_id)

        self._recompute_global_bound(task_id, START, LB)
        self._recompute_global_bound(task_id, END, LB)

        if feasible_machines:
            self.event_queue.append(DomainEvent(task_id, START_LB))

    def set_machine_start_ub(
        self, task_id: TaskID, ub: Time, machine_id: MachineID
    ) -> None:
        idx = task_id * self.start.n_machines + machine_id

        old_ub = self.start.ubs[idx]
        self.start.ubs[idx] = ub

        if ub < self.start.lbs[idx] or self.end.ubs[idx] < self.end.lbs[idx]:
            self.restrict_machine(task_id, machine_id)
            return

        if old_ub == self.start.global_ubs[task_id]:
            self._recompute_global_bound(task_id, START, UB)
            self._recompute_global_bound(task_id, END, UB)

        self.event_queue.append(DomainEvent(task_id, START_UB, machine_id))

    def set_start_ub(self, task_id: TaskID, ub: Time) -> None:
        row = task_id * self.start.n_machines
        feasible_machines = self.feasible_machines[task_id]

        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_ubs = self.end.ubs
        remaining_times = self.remaining_times

        for m_id in feasible_machines[:]:
            idx = row + m_id

            if start_ubs[idx] > ub:
                start_ubs[idx] = ub
                end_ubs[idx] = ub + remaining_times[idx]

                if ub < start_lbs[idx] or end_ubs[idx] < self.end.lbs[idx]:
                    self.restrict_machine(task_id, m_id)

        self._recompute_global_bound(task_id, START, UB)
        self._recompute_global_bound(task_id, END, UB)

        if feasible_machines:
            self.event_queue.append(DomainEvent(task_id, START_UB))

    def set_machine_end_lb(
        self, task_id: TaskID, lb: Time, machine_id: MachineID
    ) -> None:
        idx = task_id * self.start.n_machines + machine_id

        old_lb = self.end.lbs[idx]
        start_lb = lb - self.remaining_times[idx]

        self.end.lbs[idx] = lb
        self.start.lbs[idx] = start_lb

        if lb > self.end.ubs[idx] or self.start.lbs[idx] > self.start.ubs[idx]:
            self.restrict_machine(task_id, machine_id)
            return

        if old_lb == self.end.global_lbs[task_id]:
            self._recompute_global_bound(task_id, END, LB)
            self._recompute_global_bound(task_id, START, LB)

        self.event_queue.append(DomainEvent(task_id, END_LB, machine_id))

    def set_end_lb(self, task_id: TaskID, lb: Time) -> None:
        row = task_id * self.start.n_machines
        feasible_machines = self.feasible_machines[task_id]

        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_lbs = self.end.lbs
        end_ubs = self.end.ubs
        remaining_times = self.remaining_times

        for m_id in feasible_machines[:]:
            idx = row + m_id

            if end_lbs[idx] < lb:
                end_lbs[idx] = lb
                derived_start_lb = lb - remaining_times[idx]
                if start_lbs[idx] < derived_start_lb:
                    start_lbs[idx] = derived_start_lb

                if lb > end_ubs[idx] or start_lbs[idx] > start_ubs[idx]:
                    self.restrict_machine(task_id, m_id)

        self._recompute_global_bound(task_id, END, LB)
        self._recompute_global_bound(task_id, START, LB)

        if feasible_machines:
            self.event_queue.append(DomainEvent(task_id, END_LB))

    def set_machine_end_ub(
        self, task_id: TaskID, ub: Time, machine_id: MachineID
    ) -> None:
        idx = task_id * self.start.n_machines + machine_id

        old_ub = self.end.ubs[idx]
        start_ub = ub - self.remaining_times[idx]

        self.end.ubs[idx] = ub
        self.start.ubs[idx] = min(self.start.ubs[idx], start_ub)

        if ub < self.end.lbs[idx] or self.start.ubs[idx] < self.start.lbs[idx]:
            self.restrict_machine(task_id, machine_id)
            return

        if old_ub == self.end.global_ubs[task_id]:
            self._recompute_global_bound(task_id, END, UB)
            self._recompute_global_bound(task_id, START, UB)

        self.event_queue.append(DomainEvent(task_id, END_UB, machine_id))

    def set_end_ub(self, task_id: TaskID, ub: Time) -> None:
        row = task_id * self.start.n_machines
        feasible_machines = self.feasible_machines[task_id]

        start_lbs = self.start.lbs
        start_ubs = self.start.ubs
        end_lbs = self.end.lbs
        end_ubs = self.end.ubs
        remaining_times = self.remaining_times

        for m_id in feasible_machines[:]:
            idx = row + m_id

            if end_ubs[idx] > ub:
                end_ubs[idx] = ub
                derived_start_ub = ub - remaining_times[idx]
                if start_ubs[idx] > derived_start_ub:
                    start_ubs[idx] = derived_start_ub

                if ub < end_lbs[idx] or start_ubs[idx] < start_lbs[idx]:
                    self.restrict_machine(task_id, m_id)

        self._recompute_global_bound(task_id, END, UB)
        self._recompute_global_bound(task_id, START, UB)

        if feasible_machines:
            self.event_queue.append(DomainEvent(task_id, END_UB))

    def assign(
        self, task_id: TaskID, time: Time, machine_id: MachineID
    ) -> None:
        row = task_id * self.start.n_machines
        idx = row + machine_id
        duration = self.remaining_times[idx]

        end_time = time + duration

        start = self.start
        end = self.end
        presence = self.presence

        if (
            time < start.lbs[idx]
            or time > start.ubs[idx]
            or end_time < end.lbs[idx]
            or end_time > end.ubs[idx]
        ):
            raise RuntimeError(
                f"Cannot assign task {task_id} to machine {machine_id} at time {time}, "
                f"it violates the bounds for that machine: "
                f"start interval = [{start.lbs[idx]}, {start.ubs[idx]}]."
            )

        elif not can_be_present(presence[task_id]):
            raise RuntimeError(
                f"Cannot assign task {task_id} to machine {machine_id} at time {time}, "
                f"it violates the presence constraints for that task: "
                f"presence = {presence_to_str(self.presence[task_id])}."
            )

        self.assignment[task_id] = machine_id
        self.fixed[task_id] = True
        self.feasible_machines[task_id].clear()

        start.lbs[idx] = time
        start.ubs[idx] = time
        end.lbs[idx] = end_time
        end.ubs[idx] = end_time
        presence[task_id] = PRESENT

        start.global_lbs[task_id] = time
        start.global_ubs[task_id] = time
        end.global_lbs[task_id] = end_time
        end.global_ubs[task_id] = end_time

        self.event_queue.append(DomainEvent(task_id, ASSIGNMENT, machine_id))

        self.awaiting_tasks.remove(task_id)

    def pause(self, task_id: TaskID, time: Time) -> None:
        if not self.fixed[task_id]:
            raise RuntimeError(
                f"Cannot pause task {task_id} at {time}, the task was never "
                f"assigned to a machine."
            )

        start = self.start
        end = self.end

        expected_end = end.global_ubs[task_id]
        if time >= expected_end:
            raise RuntimeError(
                f"Cannot pause task {task_id} at {time}, the task has already"
                f"finished"
            )

        task_start = start.global_lbs[task_id]
        expected_duration = expected_end - task_start
        actual_duration = time - task_start

        remaining_times = self.remaining_times
        prev_assignment = self.assignment[task_id]

        n_machines = self.end.n_machines
        row = task_id * n_machines
        original_machines = self.original_machines[task_id]
        for m_id in original_machines:
            idx = row + m_id

            work_done = ((actual_duration) * remaining_times[idx]) // (
                expected_duration
            )
            remaining_times[idx] -= work_done

            start.lbs[idx] = time
            start.ubs[idx] = MAX_TIME - remaining_times[idx]
            end.lbs[idx] = time + remaining_times[idx]
            end.ubs[idx] = MAX_TIME

        start.global_lbs[task_id] = time
        self._recompute_global_bound(task_id, START, UB)

        self._recompute_global_bound(task_id, END, LB)
        end.global_ubs[task_id] = MAX_TIME

        self.assignment[task_id] = GLOBAL_MACHINE_ID
        self.fixed[task_id] = False
        self.feasible_machines[task_id].extend(original_machines)

        self.event_queue.append(DomainEvent(task_id, PAUSE, prev_assignment))

    def reset_bounds(
        self, task_id: TaskID, time: Time, machine_id: MachineID
    ) -> None:
        if self.fixed[task_id] or not can_be_present(self.presence[task_id]):
            return None

        start = self.start
        end = self.end

        n_machines = self.end.n_machines
        row = task_id * n_machines

        if machine_id != GLOBAL_MACHINE_ID:
            idx = row + machine_id
            remaining_time = self.remaining_times[idx]

            start.lbs[idx] = time
            start.ubs[idx] = MAX_TIME - remaining_time
            end.lbs[idx] = time + remaining_time
            end.ubs[idx] = MAX_TIME

            start.global_lbs[task_id] = time
            self._recompute_global_bound(task_id, START, UB)

            end.global_ubs[task_id]

        original_machines = self.original_machines[task_id]
        remaining_times = self.remaining_times

        for m_id in original_machines:
            idx = row + m_id

            start.lbs[idx] = time
            start.ubs[idx] = MAX_TIME - remaining_times[idx]
            end.lbs[idx] = time + remaining_times[idx]
            end.ubs[idx] = MAX_TIME

        start.global_lbs[task_id] = time
        self._recompute_global_bound(task_id, START, UB)

        self._recompute_global_bound(task_id, END, LB)
        end.global_ubs[task_id] = MAX_TIME

        self.feasible_machines[task_id].clear()
        self.feasible_machines[task_id].extend(original_machines)

        self.event_queue.append(DomainEvent(task_id, BOUNDS_RESET))

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
            self.remaining_times == value.remaining_times
            and self.assignment == value.assignment
            and self.presence == value.presence
            and self.start == value.start
            and self.end == value.end
        )

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.remaining_times,
            self.assignment,
            self.presence,
            self.start,
            self.end,
            self.feasible_machines,
            self.fixed,
            self.event_queue,
            self.awaiting_tasks
        )
        return (self.__class__, (DUMMY_INSTANCE,), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.remaining_times,
            self.assignment,
            self.presence,
            self.start,
            self.end,
            self.feasible_machines,
            self.fixed,
            self.event_queue,
            self.awaiting_tasks
        ) = state

from typing import Any, TypeAlias, Literal, cast

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    MachineID, TaskID, Time, Status,
    MIN_TIME, MAX_TIME, GLOBAL_MACHINE_ID,
    EzPickle
)

from cpscheduler.environment.state.csp import (
    TaskDomains,
    Presence,
    PresenceType,
    presence_to_str
)

from cpscheduler.environment.state.events import (
    DomainEvent,
    RuntimeEvent,
    VarField,
    VarFieldType,
    RuntimeEventKind
)

from cpscheduler.environment.state.instance import ProblemInstance
from cpscheduler.environment.state.runtime import RuntimeState, TaskHistory

# import cpscheduler.environment.debug as debug

ObsType: TypeAlias = tuple[dict[str, list[Any]], dict[str, list[Any]]]

PRESENT = Presence.PRESENT
ABSENT = Presence.ABSENT
INFEASIBLE = Presence.INFEASIBLE

def can_be_present(presence: PresenceType) -> bool:
    return (presence & PRESENT) != 0

def can_be_absent(presence: PresenceType) -> bool:
    return (presence & ABSENT) != 0

AWAITING = Status.AWAITING
PAUSED = Status.PAUSED
EXECUTING = Status.EXECUTING
COMPLETED = Status.COMPLETED

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

TASK_STARTED = RuntimeEventKind.TASK_STARTED
TASK_PAUSED = RuntimeEventKind.TASK_PAUSED
TASK_COMPLETED = RuntimeEventKind.TASK_COMPLETED


DUMMY_INSTANCE = ProblemInstance({})

IntervalEnd = bool
START: IntervalEnd = True
END: IntervalEnd = False

Bound = bool
LB: Bound = True
UB: Bound = False

UNKNOWN_TASK: TaskID = -1

@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class ScheduleState(EzPickle):
    """
    ScheduleState represents the current state of the scheduling environment,
    working as both a Discrete Event Simulation (DES) state and a Constraint
    Satisfaction Problem (CSP) state.

    It has no simulation logic itself, instead, it provides an API to read and
    modify the current state of the problem. The actual kernel is implemented in
    the SchedulingEnv class, which knows the constraints and how to propagate
    changes through them.
    """

    instance: ProblemInstance

    time: Time

    infeasible: bool

    domains: TaskDomains

    runtime: RuntimeState

    domain_event_queue: list[DomainEvent]
    runtime_event_queue: list[RuntimeEvent]

    _debug_checks: bool

    def __init__(self) -> None:
        self.instance = ProblemInstance({}) # Dummy instance

        self.time = 0

        self.infeasible = False
        self._debug_checks = False

        self.domain_event_queue = []
        self.runtime_event_queue = []

    def set_debug_checks(self, enabled: bool = True) -> None:
        self._debug_checks = bool(enabled)

    def _debug_validate_machine_id(
        self,
        task_id: TaskID,
        machine_id: MachineID,
        allow_global: bool = False,
        origin: str = "_debug_validate_machine_id"
    ) -> None:
        if allow_global and machine_id == GLOBAL_MACHINE_ID:
            return

        if machine_id < 0 or machine_id >= self.n_machines:
            raise RuntimeError(
                f"{origin}: Invalid machine_id={machine_id} for task {task_id}. "
                f"Expected [0, {self.n_machines - 1}]."
            )

        if machine_id not in self.instance.processing_times[task_id]:
            raise RuntimeError(
                f"{origin}: Machine {machine_id} is not valid for task {task_id}."
            )

    def _debug_validate_bounds(
        self,
        task_id: TaskID,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
        origin: str = "_debug_validate_bounds"
    ) -> None:
        domains = self.domains
        row = task_id * self.n_machines

        machines: list[MachineID]
        if machine_id == GLOBAL_MACHINE_ID:
            machines = list(domains.feasible_machines[task_id])
        else:
            self._debug_validate_machine_id(task_id, machine_id)
            machines = [machine_id]

        for m_id in machines:
            idx = row + m_id

            start_lb = domains.start.lbs[idx]
            start_ub = domains.start.ubs[idx]
            end_lb = domains.end.lbs[idx]
            end_ub = domains.end.ubs[idx]
            remaining = domains.remaining_times[idx]

            if start_lb > start_ub:
                raise RuntimeError(
                    f"Invalid start bounds for task {task_id} on machine {m_id}: "
                    f"[{start_lb}, {start_ub}]."
                )

            if end_lb > end_ub:
                raise RuntimeError(
                    f"Invalid end bounds for task {task_id} on machine {m_id}: "
                    f"[{end_lb}, {end_ub}]."
                )

            if start_lb + remaining > end_ub:
                raise RuntimeError(
                    f"Inconsistent bounds for task {task_id} on machine {m_id}: "
                    f"start_lb({start_lb}) + p({remaining}) > end_ub({end_ub})."
                )

            if end_lb - remaining > start_ub:
                raise RuntimeError(
                    f"Inconsistent bounds for task {task_id} on machine {m_id}: "
                    f"end_lb({end_lb}) - p({remaining}) > start_ub({start_ub})."
                )

    # Properties
    @property
    def n_machines(self) -> int:
        return self.instance.n_machines

    @property
    def n_tasks(self) -> int:
        return self.instance.n_tasks

    @property
    def n_jobs(self) -> int:
        return self.instance.n_jobs

    @property
    def loaded(self) -> bool:
        return self.instance.loaded

    @property
    def debug_mode(self) -> bool:
        return self._debug_checks

    # Instance control methods
    def read_instance(
        self,
        task_data: dict[str, list[Any]],
    ) -> None:
        self.instance = ProblemInstance(task_data)

    # Flow control methods
    def clear(self) -> None:
        self.time = 0
        self.infeasible = False
        self.instance = ProblemInstance({})
        self.domains = TaskDomains(self.instance)
        self.runtime = RuntimeState(self.instance)
        self.domain_event_queue.clear()
        self.runtime_event_queue.clear()

    def reset(self) -> None:
        assert (
            self.loaded
        ), "No instance loaded. Please load an instance before resetting the state."

        self.time = 0
        self.infeasible = False

        self.domains = TaskDomains(self.instance)
        self.runtime = RuntimeState(self.instance)

        for task_id in range(self.n_tasks):
            self._recompute_global_bound(task_id, END, LB)
            self._recompute_global_bound(task_id, START, UB)

        self.domain_event_queue.clear()
        self.runtime_event_queue.clear()

    def is_terminal(self) -> bool:
        if self.infeasible:
            return True

        awaiting = self.runtime.awaiting_tasks
        executing = self.runtime.executing_tasks

        return not awaiting and not executing

    def advance_time_(self, new_time: Time) -> None:
        assert (
            new_time > self.time
        ), "Advance time must be monotonic increasing."

        self.time = new_time

        runtime = self.runtime
        executing_tasks = runtime.get_executing_tasks()
        for task_id in executing_tasks:
            assignment = runtime.get_assignment(task_id)
            end_time = runtime.get_end(task_id)

            if end_time <= new_time:
                runtime.executing_tasks.remove(task_id)
                runtime.completed_tasks.add(task_id)

                runtime.status[task_id] = COMPLETED
                self.runtime_event_queue.append(
                    RuntimeEvent(task_id, TASK_COMPLETED, assignment)
                )

    # Constraint propagation API methods

    ## Getter methods for variable values
    def get_start_lb(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self.domains.start.get_lb(task_id, machine_id)

    def get_start_ub(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self.domains.start.get_ub(task_id, machine_id)

    def get_end_lb(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self.domains.end.get_lb(task_id, machine_id)

    def get_end_ub(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self.domains.end.get_ub(task_id, machine_id)

    def get_remaining_time(
        self, task_id: TaskID, machine_id: MachineID
    ) -> Time:
        idx = task_id * self.instance.n_machines + machine_id

        return self.domains.remaining_times[idx]

    def get_assignment(self, task_id: TaskID) -> MachineID:
        return self.domains.assignment[task_id]

    def get_machines(self, task_id: TaskID) -> tuple[MachineID, ...]:
        return self.domains.get_feasible_machines(task_id)

    def is_fixed(self, task_id: TaskID) -> bool:
        return self.domains.assignment[task_id] != GLOBAL_MACHINE_ID

    def is_present(self, task_id: TaskID) -> bool:
        return self.domains.presence[task_id] == PRESENT

    def is_absent(self, task_id: TaskID) -> bool:
        return self.domains.presence[task_id] == ABSENT

    def is_feasible(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:

        if self.domains.presence[task_id] == INFEASIBLE:
            return False

        machines = self.domains.feasible_machines[task_id]

        if machine_id == GLOBAL_MACHINE_ID:
            return bool(machines)

        return machine_id in machines

    ## Setter methods for variable values, triggering constraint propagation through events
    def _recompute_global_bound(
        self, task_id: TaskID, var: IntervalEnd, bound: Bound
    ) -> None:
        domains = self.domains

        variable = domains.start if var else domains.end
        row = task_id * variable.n_machines

        feasible_machines = domains.feasible_machines[task_id]

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

    def _recompute_all_bounds(self, task_id: TaskID) -> None:
        self._recompute_global_bound(task_id, START, LB)
        self._recompute_global_bound(task_id, START, UB)
        self._recompute_global_bound(task_id, END, LB)
        self._recompute_global_bound(task_id, END, UB)

    def _restrict_presence(self, task_id: TaskID, mask: Literal[0b01, 0b10]) -> None:
        domains = self.domains
        runtime = self.runtime

        old_presence = domains.presence[task_id]
        # Bitwise operations on Literal unions are inferred as int by type checkers.
        # Explicitly narrow back to PresenceType.
        new_presence = cast(PresenceType, old_presence & mask)

        field: VarFieldType
        if new_presence == old_presence:
            return

        if new_presence == INFEASIBLE:
            domains.presence[task_id] = INFEASIBLE
            self.fail(task_id)
            return

        if new_presence == PRESENT:
            field = PRESENCE

        elif new_presence == ABSENT:
            runtime.awaiting_tasks.discard(task_id)
            runtime.unlocked_tasks.discard(task_id)
            field = ABSENCE

        else:
            raise RuntimeError(
                f"Unreachable: unexpected presence value {new_presence!r}"
            )

        domains.presence[task_id] = new_presence

        self.domain_event_queue.append(DomainEvent(task_id, field))

    def require_task(self, task_id: TaskID) -> None:
        self._restrict_presence(task_id, PRESENT)

    def forbid_task(self, task_id: TaskID) -> None:
        self._restrict_presence(task_id, ABSENT)

    def restrict_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        domains = self.domains

        feasible_machines = domains.feasible_machines[task_id]

        if machine_id not in feasible_machines:
            return

        feasible_machines.remove(machine_id)

        if not feasible_machines:
            self._restrict_presence(task_id, ABSENT)
            return

        self._recompute_all_bounds(task_id)

        self.domain_event_queue.append(
            DomainEvent(task_id, MACHINE_INFEASIBLE, machine_id)
        )

    def tight_start_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        domains = self.domains

        if value <= domains.start.get_lb(task_id, machine_id):
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        remaining_times = domains.remaining_times

        if machine_id == GLOBAL_MACHINE_ID:
            row = task_id * self.n_machines

            feasible_machines = domains.get_feasible_machines(task_id)
            for m_id in feasible_machines:
                idx = row + m_id

                if start_lbs[idx] < value:
                    start_lbs[idx] = value
                    end_lbs[idx] = value + remaining_times[idx]

                    if value > start_ubs[idx] or end_lbs[idx] > end_ubs[idx]:
                        self.restrict_machine(task_id, m_id)

            self._recompute_global_bound(task_id, START, LB)
            self._recompute_global_bound(task_id, END, LB)

            if feasible_machines:
                self.domain_event_queue.append(
                    DomainEvent(task_id, START_LB)
                )

            return

        idx = task_id * self.n_machines + machine_id

        old_lb = start_lbs[idx]
        end_lb = value + remaining_times[idx]

        start_lbs[idx] = value
        end_lbs[idx] = end_lb

        if value > start_ubs[idx] or end_lb > end_ubs[idx]:
            self.restrict_machine(task_id, machine_id)
            return

        if old_lb == domains.start.global_lbs[task_id]:
            self._recompute_global_bound(task_id, START, LB)
            self._recompute_global_bound(task_id, END, LB)


        self.domain_event_queue.append(
            DomainEvent(task_id, START_LB, machine_id)
        )

    def tight_start_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        domains = self.domains

        if value >= domains.start.get_ub(task_id, machine_id):
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_ubs = domains.end.ubs
        remaining_times = domains.remaining_times

        if machine_id == GLOBAL_MACHINE_ID:
            row = task_id * self.n_machines

            feasible_machines = domains.get_feasible_machines(task_id)
            for m_id in feasible_machines:
                idx = row + m_id

                if start_ubs[idx] > value:
                    start_ubs[idx] = value
                    end_ubs[idx] = value + remaining_times[idx]

                    if value < start_lbs[idx] or end_ubs[idx] < domains.end.lbs[idx]:
                        self.restrict_machine(task_id, m_id)

            self._recompute_global_bound(task_id, START, UB)
            self._recompute_global_bound(task_id, END, UB)

            if feasible_machines:
                self.domain_event_queue.append(
                    DomainEvent(task_id, START_UB)
                )

            return

        idx = task_id * self.n_machines + machine_id

        old_ub = start_ubs[idx]
        start_ubs[idx] = value
        end_ubs[idx] = value + remaining_times[idx]

        if value < start_lbs[idx] or end_ubs[idx] < domains.end.lbs[idx]:
            self.restrict_machine(task_id, machine_id)
            return

        if old_ub == domains.start.global_ubs[task_id]:
            self._recompute_global_bound(task_id, START, UB)
            self._recompute_global_bound(task_id, END, UB)

        self.domain_event_queue.append(
            DomainEvent(task_id, START_UB, machine_id)
        )


    def tight_end_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        domains = self.domains

        if value <= domains.end.get_lb(task_id, machine_id):
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        remaining_times = domains.remaining_times

        if machine_id == GLOBAL_MACHINE_ID:
            row = task_id * self.n_machines

            feasible_machines = domains.get_feasible_machines(task_id)
            for m_id in feasible_machines:
                idx = row + m_id

                if end_lbs[idx] < value:
                    end_lbs[idx] = value
                    derived_start_lb = value - remaining_times[idx]
                    if start_lbs[idx] < derived_start_lb:
                        start_lbs[idx] = derived_start_lb

                    if value > end_ubs[idx] or start_lbs[idx] > start_ubs[idx]:
                        self.restrict_machine(task_id, m_id)

            self._recompute_global_bound(task_id, END, LB)
            self._recompute_global_bound(task_id, START, LB)

            if feasible_machines:
                self.domain_event_queue.append(
                    DomainEvent(task_id, END_LB)
                )

            return

        idx = task_id * self.n_machines + machine_id

        old_lb = end_lbs[idx]
        start_lb = value - remaining_times[idx]

        end_lbs[idx] = value
        start_lbs[idx] = start_lb

        if value > end_ubs[idx] or start_lb > start_ubs[idx]:
            self.restrict_machine(task_id, machine_id)
            return

        if old_lb == domains.end.global_lbs[task_id]:
            self._recompute_global_bound(task_id, END, LB)
            self._recompute_global_bound(task_id, START, LB)



        self.domain_event_queue.append(
            DomainEvent(task_id, END_LB, machine_id)
        )


    def tight_end_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        domains = self.domains

        if value >= domains.end.get_ub(task_id, machine_id):
            return

        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        remaining_times = domains.remaining_times

        if machine_id == GLOBAL_MACHINE_ID:
            row = task_id * self.n_machines

            feasible_machines = domains.get_feasible_machines(task_id)
            for m_id in feasible_machines:
                idx = row + m_id

                if end_ubs[idx] > value:
                    end_ubs[idx] = value
                    derived_start_ub = value - remaining_times[idx]
                    if start_ubs[idx] > derived_start_ub:
                        start_ubs[idx] = derived_start_ub

                    if value < end_lbs[idx] or start_ubs[idx] < domains.start.lbs[idx]:
                        self.restrict_machine(task_id, m_id)

            self._recompute_global_bound(task_id, END, UB)
            self._recompute_global_bound(task_id, START, UB)

            if feasible_machines:
                self.domain_event_queue.append(
                    DomainEvent(task_id, END_UB)
                )

            return

        idx = task_id * self.n_machines + machine_id

        old_ub = end_ubs[idx]
        derived_start_ub = value - remaining_times[idx]

        end_ubs[idx] = value
        if start_ubs[idx] > derived_start_ub:
            start_ubs[idx] = derived_start_ub

        if value < end_lbs[idx] or start_ubs[idx] < domains.start.lbs[idx]:
            self.restrict_machine(task_id, machine_id)
            return

        if old_ub == domains.end.global_ubs[task_id]:
            self._recompute_global_bound(task_id, END, UB)
            self._recompute_global_bound(task_id, START, UB)

        self.domain_event_queue.append(
            DomainEvent(task_id, END_UB, machine_id)
        )

    def forbid_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        self.restrict_machine(task_id, machine_id)


    def require_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        for other_machine in self.instance.processing_times[task_id]:
            if other_machine != machine_id:
                self.forbid_machine(task_id, other_machine)


    def reset_bounds(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> None:
        domains = self.domains
        time = self.time

        if self._debug_checks:
            self._debug_validate_machine_id(
                task_id, machine_id,
                allow_global=True,
                origin="reset_bounds"
            )

        if (
            self.is_fixed(task_id)
            or not can_be_present(domains.presence[task_id])
        ):
            return

        start = domains.start
        end = domains.end

        remaining_times = domains.remaining_times
        feasible_machines = domains.feasible_machines[task_id]

        row = task_id * self.n_machines
        if machine_id != GLOBAL_MACHINE_ID:
            idx = row + machine_id
            remaining_time = remaining_times[idx]

            start.lbs[idx] = time
            start.ubs[idx] = MAX_TIME - remaining_time
            end.lbs[idx] = time + remaining_time
            end.ubs[idx] = MAX_TIME

            feasible_machines.add(machine_id)

            start.global_lbs[task_id] = time
            self._recompute_global_bound(task_id, START, UB)

            self._recompute_global_bound(task_id, END, LB)
            end.global_ubs[task_id] = MAX_TIME

            if self._debug_checks:
                self._debug_validate_bounds(
                    task_id, machine_id,
                    origin="reset_bounds"
                )

            return

        original_machines = domains.original_machines[task_id]
        for m_id in original_machines:
            idx = row + m_id

            start.lbs[idx] = time
            start.ubs[idx] = MAX_TIME - remaining_times[idx]
            end.lbs[idx] = time + remaining_times[idx]
            end.ubs[idx] = MAX_TIME

            feasible_machines.add(m_id)

        start.global_lbs[task_id] = time
        self._recompute_global_bound(task_id, START, UB)

        self._recompute_global_bound(task_id, END, LB)
        end.global_ubs[task_id] = MAX_TIME

        self.domain_event_queue.append(
            DomainEvent(task_id, BOUNDS_RESET)
        )

        if self._debug_checks:
            self._debug_validate_bounds(
                task_id,
                origin="reset_bounds"
            )

    def fail(self, task_id: TaskID = UNKNOWN_TASK) -> None:
        """
        Marks the current state as infeasible.

        In this CSP kernel, propagation is synchronous and domain updates are applied
        atomically. As a design invariant, all inconsistencies are expected to be
        reducible to domain contradictions (i.e., domain wipeout or forbidden
        assignments) through propagation or at assignment time.

        Therefore, well-formed propagators should express violations via domain
        reductions on individual tasks, making explicit failure signaling typically
        unnecessary.

        This method is reserved for cases where a constraint detects an inconsistency
        that cannot be reduced to any single task's domain under the current
        propagation model, or as a defensive safeguard during development.
        """
        self.infeasible = True
        self.runtime.awaiting_tasks.clear()
        self.runtime.unlocked_tasks.clear()

        self.domain_event_queue.append(
            DomainEvent(task_id, STATE_INFEASIBLE)
        )

    # Discrete event simulation API methods

    ## Getter methods for variable values
    def is_awaiting(self, task_id: TaskID) -> bool:
        return task_id in self.runtime.awaiting_tasks

    def is_available(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:
        if task_id not in self.runtime.unlocked_tasks:
            return False

        t = self.time
        start = self.domains.start

        if machine_id == GLOBAL_MACHINE_ID:
            row = task_id * self.n_machines

            for machine in self.domains.feasible_machines[task_id]:
                idx = row + machine

                if t >= start.lbs[idx] and t < start.ubs[idx]:
                    return True

            return False

        if machine_id not in self.domains.feasible_machines[task_id]:
            return False

        idx = task_id * self.n_machines + machine_id
        lb = start.lbs[idx]
        if t < lb:
            return False

        return t < start.ubs[idx]

    def is_paused(self, task_id: TaskID) -> bool:
        return self.runtime.status[task_id] == PAUSED

    def is_executing(self, task_id: TaskID) -> bool:
        return self.runtime.status[task_id] == EXECUTING

    def is_completed(self, task_id: TaskID) -> bool:
        return self.runtime.status[task_id] == COMPLETED

    def is_locked(self, task_id: TaskID) -> bool:
        return bool(self.runtime.prerequisites[task_id])

    def get_end(self, task_id: TaskID) -> Time:
        if not self.is_fixed(task_id):
            raise RuntimeError(f"Task {task_id} has not been commited yet.")

        return self.runtime.get_end(task_id)

    def get_start(self, task_id: TaskID) -> Time:
        if not self.is_fixed(task_id):
            raise RuntimeError(f"Task {task_id} has not been commited yet.")

        return self.runtime.get_start(task_id)

    ## Setter methods for variable values, triggering constraint propagation through events
    def add_prerequisite(self, task_id: TaskID, name: str) -> None:
        self.runtime.prerequisites[task_id].add(name)
        self.runtime.unlocked_tasks.discard(task_id)

    def satisfy_prerequisite(self, task_id: TaskID, name: str) -> None:
        prerequisites = self.runtime.prerequisites[task_id]

        prerequisites.discard(name)
        if not prerequisites:
            self.runtime.unlocked_tasks.add(task_id)

    def execute_task(self, task_id: TaskID, machine_id: MachineID) -> None:
        if machine_id == GLOBAL_MACHINE_ID:
            raise ValueError(f"Cannot assign to the global machine {GLOBAL_MACHINE_ID}.")

        start_time = self.time
        domains = self.domains
        runtime = self.runtime

        if not self.is_available(task_id, machine_id):
            raise RuntimeError(
                f"Cannot assign task {task_id} to machine {machine_id} at time "
                f"{start_time}, the task is not available at the current time."
            )

        feasible_machines = domains.feasible_machines[task_id]
        if machine_id not in feasible_machines:
            lb = domains.start.get_lb(task_id, machine_id)
            ub = domains.start.get_ub(task_id, machine_id)

            raise RuntimeError(
                f"Cannot assign task {task_id} to machine {machine_id} at time "
                f"{start_time}, because this machine is not feasible. "
                f"Start Interval = [{lb}, {ub}]."
            )

        presence = domains.presence[task_id]
        if not can_be_present(presence):
            raise RuntimeError(
                f"Cannot assign task {task_id} to machine {machine_id} at time "
                f"{start_time}, it violates the presence constraints for that "
                f"task: presence = {presence_to_str(presence)}."
            )

        row = task_id * self.n_machines
        idx = row + machine_id
        duration = domains.remaining_times[idx]

        end_time = start_time + duration

        start = domains.start
        end = domains.end

        domains.assignment[task_id] = machine_id
        domains.feasible_machines[task_id].clear()
        domains.feasible_machines[task_id].add(machine_id)
        domains.presence[task_id] = PRESENT

        start.lbs[idx] = start_time
        start.ubs[idx] = start_time
        end.lbs[idx] = end_time
        end.ubs[idx] = end_time

        start.global_lbs[task_id] = start_time
        start.global_ubs[task_id] = start_time
        end.global_lbs[task_id] = end_time
        end.global_ubs[task_id] = end_time

        self.domain_event_queue.append(
            DomainEvent(task_id, ASSIGNMENT, machine_id)
        )
        self.runtime_event_queue.append(
            RuntimeEvent(task_id, TASK_STARTED, machine_id)
        )

        if end_time > runtime.last_completion_time:
            runtime.last_completion_time = end_time

        runtime.unlocked_tasks.remove(task_id)
        runtime.awaiting_tasks.remove(task_id)
        runtime.executing_tasks.add(task_id)
        runtime.status[task_id] = EXECUTING

        runtime.history[task_id].append(
            TaskHistory(machine_id, start_time, end_time)
        )

        if self._debug_checks:
            self._debug_validate_bounds(
                task_id, machine_id,
                origin="execute_task"
            )

    def pause_task(self, task_id: TaskID) -> None:
        pause_time = self.time

        if not self.is_executing(task_id):
            raise RuntimeError(
                f"Cannot pause task {task_id} at {pause_time}, the task is not "
                f"currently executing."
            )
        
        domains = self.domains
        runtime = self.runtime

        start = domains.start
        end = domains.end

        expected_end = end.global_ubs[task_id]
        task_start = start.global_lbs[task_id]

        expected_duration = expected_end - task_start
        actual_duration = pause_time - task_start

        remaining_times = domains.remaining_times
        prev_assignment = domains.assignment[task_id]

        row = task_id * self.n_machines
        original_machines = domains.original_machines[task_id]
        feasible_machines = domains.feasible_machines[task_id]

        if expected_duration == 0:
            expected_duration = 1

        for m_id in original_machines:
            idx = row + m_id

            work_done = ((actual_duration) * remaining_times[idx]) // (
                expected_duration
            )
            remaining_times[idx] -= work_done

            start.lbs[idx] = pause_time
            start.ubs[idx] = MAX_TIME - remaining_times[idx]
            end.lbs[idx] = pause_time + remaining_times[idx]
            end.ubs[idx] = MAX_TIME

            feasible_machines.add(m_id)

        start.global_lbs[task_id] = pause_time
        self._recompute_global_bound(task_id, START, UB)

        self._recompute_global_bound(task_id, END, LB)
        end.global_ubs[task_id] = MAX_TIME

        domains.assignment[task_id] = GLOBAL_MACHINE_ID

        self.domain_event_queue.append(
            DomainEvent(task_id, PAUSE, prev_assignment)
        )

        runtime.executing_tasks.remove(task_id)
        runtime.awaiting_tasks.add(task_id)
        runtime.unlocked_tasks.add(task_id)

        history = runtime.history[task_id]

        prev_entry = history.pop()
        history.append(
            TaskHistory(
                prev_entry.machine_id, prev_entry.start_time, pause_time
            )
        )

        runtime.status[task_id] = PAUSED
        self.runtime_event_queue.append(
            RuntimeEvent(task_id, TASK_PAUSED, prev_entry.machine_id)
        )

        if prev_entry.end_time == runtime.last_completion_time:
            runtime.recompute_last_completion_time()

        if self._debug_checks:
            self._debug_validate_bounds(
                task_id,
                origin="pause_task"
            )

    # Runtime utils

    def get_next_start_lb(self) -> Time:
        min_lb = MAX_TIME

        global_lbs = self.domains.start.global_lbs
        unlocked_tasks = self.runtime.unlocked_tasks
        current_time = self.time

        for task_id in unlocked_tasks:
            lb = global_lbs[task_id]

            if lb <= current_time:
                return current_time

            if lb < min_lb:
                min_lb = global_lbs[task_id]

        return min_lb

    def get_last_completion_time(self) -> Time:
        return self.runtime.last_completion_time

    def get_machine_execution(self) -> dict[MachineID, list[TaskID]]:
        assignments: dict[MachineID, list[TaskID]] = {
            machine_id: [] for machine_id in range(self.instance.n_machines)
        }

        runtime = self.runtime
        for task_id in runtime.executing_tasks:
            machine_id = runtime.get_assignment(task_id)
            assignments[machine_id].append(task_id)

        return assignments

    def get_observation(self) -> ObsType:
        task_obs = self.instance.task_instance.copy()
        task_obs["status"] = self.runtime.status.copy()

        available = [False] * self.n_tasks
        for task_id in self.runtime.unlocked_tasks:
            available[task_id] = self.is_available(task_id)

        task_obs["available"] = available

        return task_obs, {}

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, ScheduleState):
            return False

        return (
            self.instance == value.instance
            and self.time == value.time
            and self.infeasible == value.infeasible
            and self._debug_checks == value._debug_checks
            and self.domains == value.domains
            and self.runtime == value.runtime
            and self.domain_event_queue == value.domain_event_queue
            and self.runtime_event_queue == value.runtime_event_queue
        )

"""Scheduling Environment State Module.

This module provides ScheduleState, the core kernel for maintaining and querying
the state of a constraint satisfaction problem (CSP) combined with discrete event
simulation (DES). The state objects coordinate between domain constraints
(e.g., task start/end time bounds, machine feasibility) and runtime simulation
(e.g., task execution, completion events).
"""

from typing import Any, Literal, cast

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    MIN_TIME,
    EzPickle,
    MachineID,
    Status,
    TaskID,
    Time,
)
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state.csp import (
    Presence,
    PresenceType,
    TaskDomains,
    presence_to_str,
)
from cpscheduler.environment.state.events import (
    DomainEvent,
    RuntimeEvent,
    RuntimeEventKind,
    VarField,
    VarFieldType,
)
from cpscheduler.environment.state.runtime import RuntimeState, TaskHistory
from cpscheduler.environment.utils.debug import (
    validate_domain_bounds,
    validate_machine_id,
)

PRESENT = Presence.PRESENT
ABSENT = Presence.ABSENT
INFEASIBLE = Presence.INFEASIBLE


def can_be_present(presence: PresenceType) -> bool:
    """Check if presence is undecided, or present."""
    return (presence & PRESENT) != 0


def can_be_absent(presence: PresenceType) -> bool:
    """Check if presence is undecided, or absent."""
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
TASK_MACHINE_INFEASIBLE = RuntimeEventKind.TASK_MACHINE_INFEASIBLE

IntervalEnd = bool
START: IntervalEnd = True
END: IntervalEnd = False

Bound = bool
LB: Bound = True
UB: Bound = False

UNKNOWN_TASK: TaskID = -1


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class ScheduleState(EzPickle):
    """Core CSP/DES state kernel for scheduling problems.

    ScheduleState maintains the constraint satisfaction problem (CSP) state
    (variable domains) and discrete event simulation (DES) state (task execution
    and completion) for a scheduling environment.
    It provides a API to read and mutate state, delegating constraint
    propagation logic to the environment via event queues.

    The state maintains two views:

    1. **CSP View (Domains)**: Represents the feasible space containing the domain variables:
        - Start/End time bounds (per task and machine)
        - Machine feasibility sets (which machines can process each task)
        - Presence bitfield (whether each task can be present, absent, or is infeasible)

    2. **DES View (Runtime)**: Represents the dynamic execution state of the schedule:
       - Task scheduling history (machine assignment, start/end times)
       - Task dependencies and lock status
       - Current task statuses (AWAITING, EXECUTING, PAUSED, COMPLETED)
    """

    instance: ProblemInstance

    time: Time

    infeasible: bool

    domains: TaskDomains

    runtime: RuntimeState

    domain_event_queue: list[DomainEvent]
    runtime_event_queue: list[RuntimeEvent]

    _debug: bool

    def __init__(self, instance: ProblemInstance) -> None:
        """Initialize the ScheduleState with a problem instance.

        Parameters
        ----------
        instance: ProblemInstance
            The problem instance containing tasks, machines, processing times, etc.

        """
        self.instance = instance

        self.time = 0

        self.infeasible = False
        self._debug = instance.debug

        self.domains = TaskDomains(instance)
        self.runtime = RuntimeState(instance)

        self.domain_event_queue = []
        self.runtime_event_queue = []

    # Properties

    @property
    def n_machines(self) -> int:
        """Return the number of machines in the problem instance."""
        return self.instance.n_machines

    @property
    def n_tasks(self) -> int:
        """Return the number of tasks in the problem instance."""
        return self.instance.n_tasks

    @property
    def n_jobs(self) -> int:
        """Return the number of jobs in the problem instance."""
        return self.instance.n_jobs

    @property
    def debug(self) -> bool:
        """Return whether debug mode is enabled for the state."""
        return self._debug

    # Flow control methods
    def reset(self) -> None:
        """Reset state to initial condition while preserving the problem instance."""
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
        """Return True if the problem is infeasible or all tasks are completed/absent."""
        return self.infeasible or (
            not self.runtime.awaiting_tasks and not self.runtime.executing_tasks
        )

    def advance_time_(self, new_time: Time) -> None:
        """Advance simulation time.

        Moves the simulation clock forward (must be monotonically increasing) and
        marks any executing tasks that have reached their planned end times as
        completed.
        Queues TASK_COMPLETED events for each completed task.

        Parameters
        ----------
        new_time : Time
            New simulation time (must be > current time).

        """
        assert new_time > self.time, (
            "Advance time must be monotonic increasing."
        )

        self.time = new_time

        runtime = self.runtime
        for task_id in self.get_executing_tasks():
            assignment = runtime.get_assignment(task_id)
            end_time = runtime.get_end(task_id)

            if end_time <= new_time:
                runtime.executing_tasks.remove(task_id)
                runtime.completed_tasks.add(task_id)

                runtime.status[task_id] = COMPLETED
                self.runtime_event_queue.append(
                    RuntimeEvent(task_id, TASK_COMPLETED, assignment)
                )

    # Problem Instance API methods

    ## Getter methods for instance parameters
    def is_preemptive(self, task_id: TaskID) -> bool:
        """Return whether a task allows preemption."""
        return self.instance.preemptive[task_id]

    def is_optional(self, task_id: TaskID) -> bool:
        """Return whether a task is optional (can be left unassigned)."""
        return self.instance.optional[task_id]

    def has_processing_time(
        self, task_id: TaskID, machine_id: MachineID
    ) -> bool:
        """Return whether a task can be processed on a given machine."""
        return self.instance.machine_mask[task_id][machine_id]

    def get_processing_time(
        self, task_id: TaskID, machine_id: MachineID
    ) -> Time:
        """Return the processing time for a task on a machine."""
        if self.has_processing_time(task_id, machine_id):
            return self.instance.processing_times[task_id][machine_id]

        raise ValueError(
            f"get_processing_time: Task {task_id} cannot be processed in Machine {machine_id}"
        )

    def get_original_machines(self, task_id: TaskID) -> list[MachineID]:
        """Return a list of all machines that can process a task."""
        return self.instance.get_machines(task_id)

    # Constraint propagation API methods

    ## Getter methods for variable values
    def get_start_lb(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        """Return the lower bound of the task start time."""
        return self.domains.start.get_lb(task_id, machine_id)

    def get_start_ub(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        """Return the upper bound of the task start time."""
        return self.domains.start.get_ub(task_id, machine_id)

    def get_end_lb(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        """Return the lower bound of the task end time."""
        return self.domains.end.get_lb(task_id, machine_id)

    def get_end_ub(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        """Return the upper bound of the task end time."""
        return self.domains.end.get_ub(task_id, machine_id)

    def get_remaining_time(
        self, task_id: TaskID, machine_id: MachineID
    ) -> Time:
        """Return the remaining processing time for a task on a machine."""
        idx = task_id * self.instance.n_machines + machine_id

        return self.domains.remaining_times[idx]

    def get_assignment(self, task_id: TaskID) -> MachineID:
        """Return the machine assigned to a task, or GLOBAL_MACHINE_ID if unassigned."""
        return self.domains.assignment[task_id]

    def get_machines(self, task_id: TaskID) -> tuple[MachineID, ...]:
        """Return the tuple of currently feasible machines for a task."""
        return self.domains.get_feasible_machines(task_id)

    def is_fixed(self, task_id: TaskID) -> bool:
        """Return whether a task have been assigned to a specific machine."""
        return self.domains.assignment[task_id] != GLOBAL_MACHINE_ID

    def is_present(self, task_id: TaskID) -> bool:
        """Return whether a task is required to execute."""
        return self.domains.presence[task_id] == PRESENT

    def is_absent(self, task_id: TaskID) -> bool:
        """Return whether a task is forbidden from executing."""
        return self.domains.presence[task_id] == ABSENT

    def is_feasible(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:
        """Return whether a task is feasible executing on the given machine."""
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
        # FUTURE: This method is called very often, maybe more often than
        # needed. As this is part of a hot loop, this implementation should be
        # revisited in the future.

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
        """Recompute all four global bounds (start/end, lb/ub) for a task."""
        self._recompute_global_bound(task_id, START, LB)
        self._recompute_global_bound(task_id, START, UB)
        self._recompute_global_bound(task_id, END, LB)
        self._recompute_global_bound(task_id, END, UB)

    def _restrict_presence(
        self, task_id: TaskID, mask: Literal[0b01, 0b10]
    ) -> None:
        domains = self.domains
        runtime = self.runtime

        old_presence = domains.presence[task_id]
        # Bitwise operations on Literal unions are inferred as int by type checkers.
        # Explicitly narrow back to PresenceType.
        new_presence = cast("PresenceType", old_presence & mask)

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

            self.runtime_event_queue.append(
                RuntimeEvent(task_id, TASK_MACHINE_INFEASIBLE)
            )

            field = ABSENCE

        else:
            raise RuntimeError(
                f"Unreachable: unexpected presence value {new_presence!r}"
            )

        domains.presence[task_id] = new_presence

        self.domain_event_queue.append(DomainEvent(task_id, field))

    def require_task(self, task_id: TaskID) -> None:
        """Force a task to be present in the schedule."""
        self._restrict_presence(task_id, PRESENT)

    def forbid_task(self, task_id: TaskID) -> None:
        """Force a task to be absent in the schedule."""
        self._restrict_presence(task_id, ABSENT)

    def restrict_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Remove a machine from the feasible set for a task.

        If all machines are removed, marks the task as ABSENT.
        Otherwise, queues a MACHINE_INFEASIBLE event.
        """
        domains = self.domains

        feasible_machines = domains.feasible_machines[task_id]

        if machine_id not in feasible_machines:
            return

        feasible_machines.remove(machine_id)

        if feasible_machines:
            self._recompute_all_bounds(task_id)

            self.domain_event_queue.append(
                DomainEvent(task_id, MACHINE_INFEASIBLE, machine_id)
            )
            self.runtime_event_queue.append(
                RuntimeEvent(task_id, TASK_MACHINE_INFEASIBLE, machine_id)
            )

        else:
            self._restrict_presence(task_id, ABSENT)

    def tight_start_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        """Raise the lower bound of task start time (earliest start constraint).

        Tightens the start time lower bound, queues START_LB domain events.
        If bounds become inconsistent, removes the machine from feasible set.

        Supports global (all machines) tightening via GLOBAL_MACHINE_ID.
        """
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
                self.domain_event_queue.append(DomainEvent(task_id, START_LB))

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
        """Lower the upper bound of task start time (latest start constraint).

        Tightens the start time upper bound, queues START_UB domain events.
        If bounds become inconsistent, removes the machine from feasible set.

        Supports global (all machines) tightening via GLOBAL_MACHINE_ID.
        """
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

                    if (
                        value < start_lbs[idx]
                        or end_ubs[idx] < domains.end.lbs[idx]
                    ):
                        self.restrict_machine(task_id, m_id)

            self._recompute_global_bound(task_id, START, UB)
            self._recompute_global_bound(task_id, END, UB)

            if feasible_machines:
                self.domain_event_queue.append(DomainEvent(task_id, START_UB))

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
        """Raise the lower bound of task end time (earliest completion constraint).

        Tightens the end time lower bound, queues END_LB domain events.
        If bounds become inconsistent, removes the machine from feasible set.

        Supports global (all machines) tightening via GLOBAL_MACHINE_ID.
        """
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
                self.domain_event_queue.append(DomainEvent(task_id, END_LB))

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

        self.domain_event_queue.append(DomainEvent(task_id, END_LB, machine_id))

    def tight_end_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        """Lower the upper bound of task end time (latest completion constraint).

        Tightens the end time upper bound, queues END_UB domain events.
        If bounds become inconsistent, removes the machine from feasible set.

        Supports global (all machines) tightening via GLOBAL_MACHINE_ID.
        """
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

                    if (
                        value < end_lbs[idx]
                        or start_ubs[idx] < domains.start.lbs[idx]
                    ):
                        self.restrict_machine(task_id, m_id)

            self._recompute_global_bound(task_id, END, UB)
            self._recompute_global_bound(task_id, START, UB)

            if feasible_machines:
                self.domain_event_queue.append(DomainEvent(task_id, END_UB))

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

        self.domain_event_queue.append(DomainEvent(task_id, END_UB, machine_id))

    def forbid_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Remove a machine from the feasible set of a task."""
        self.restrict_machine(task_id, machine_id)

    def require_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Fix a task to run on a specific machine by forbidding all others."""
        for other_machine in self.domains.feasible_machines[task_id]:
            if other_machine != machine_id:
                self.forbid_machine(task_id, other_machine)

    def reset_bounds(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> None:
        """Reset start/end time bounds to allow rescheduling from current time.

        Relaxes domain bounds to [current_time, MAX_TIME] interval and adds the
        machine(s) back to feasible set.
        Used after pausing or to recover from constraint conflicts.
        """
        domains = self.domains
        time = self.time

        if self._debug:
            validate_machine_id(
                task_id,
                machine_id,
                self.instance,
                origin="reset_bounds",
                allow_global=True,
            )

        if task_id not in self.runtime.awaiting_tasks:
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

            if self._debug:
                validate_domain_bounds(
                    task_id, self, machine_id=machine_id, origin="reset_bounds"
                )

            return

        for m_id in self.instance.get_machines(task_id):
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

        self.domain_event_queue.append(DomainEvent(task_id, BOUNDS_RESET))

        if self._debug:
            validate_domain_bounds(task_id, self, origin="reset_bounds")

    def fail(self, task_id: TaskID = UNKNOWN_TASK) -> None:
        """Mark the problem as infeasible.

        Constraints should prefer domain reductions via restrict_machine or
        tight_* methods instead of this method.
        It is reserved for hard global conflicts or defensive safeguards.
        """
        self.infeasible = True
        self.runtime.awaiting_tasks.clear()
        self.runtime.unlocked_tasks.clear()

        self.domain_event_queue.append(DomainEvent(task_id, STATE_INFEASIBLE))

    # Discrete event simulation API methods

    ## Getter methods for variable values

    def get_awaiting_tasks(self) -> list[TaskID]:
        """Return a list of awaiting task IDs (unlocked, feasible, not started)."""
        return list(self.runtime.awaiting_tasks)

    def get_unlocked_tasks(self) -> list[TaskID]:
        """Return a list of unlocked task IDs (all dependencies resolved)."""
        return list(self.runtime.unlocked_tasks)

    def get_available_tasks(self) -> list[TaskID]:
        """Return a list of available task IDs (can be executed now)."""
        return [
            task_id
            for task_id in self.runtime.unlocked_tasks
            if self.is_available(task_id)
        ]

    def get_executing_tasks(self) -> list[TaskID]:
        """Return a list of currently executing task IDs."""
        return list(self.runtime.executing_tasks)

    def get_completed_tasks(self) -> list[TaskID]:
        """Return a list of completed task IDs."""
        return list(self.runtime.completed_tasks)

    def get_history(self, task_id: TaskID, segment: int = -1) -> TaskHistory:
        """Return the scheduling history entry for a task (most recent by default)."""
        return self.runtime.history[task_id][segment]

    def is_awaiting(self, task_id: TaskID) -> bool:
        """Return whether a task is in the awaiting queue."""
        return task_id in self.runtime.awaiting_tasks

    def is_available(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:
        """Return whether a task can be scheduled now on the given machine.

        A task is available if it is unlocked (all dependencies resolved) and the
        current time falls within the feasible start window [start_lb, start_ub)
        on the machine(s).
        """
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
        """Return whether a task is paused."""
        return self.runtime.status[task_id] == PAUSED

    def is_executing(self, task_id: TaskID) -> bool:
        """Return whether a task is currently executing."""
        return self.runtime.status[task_id] == EXECUTING

    def is_completed(self, task_id: TaskID) -> bool:
        """Return whether a task is completed."""
        return self.runtime.status[task_id] == COMPLETED

    def is_locked(self, task_id: TaskID) -> bool:
        """Return whether a task has unresolved dependencies."""
        return bool(self.runtime.dependencies[task_id])

    def get_end(self, task_id: TaskID) -> Time:
        """Return the end time of a fixed task; raises RuntimeError if task not assigned."""
        if not self.is_fixed(task_id):
            raise RuntimeError(f"Task {task_id} has not been commited yet.")

        return self.runtime.get_end(task_id)

    def get_start(self, task_id: TaskID) -> Time:
        """Return the start time of a fixed task; raises RuntimeError if task not assigned."""
        if not self.is_fixed(task_id):
            raise RuntimeError(f"Task {task_id} has not been commited yet.")

        return self.runtime.get_start(task_id)

    ## Setter methods for variable values, triggering constraint propagation through events
    def add_dependency(self, task_id: TaskID, name: str) -> None:
        """Add a named dependency to lock a task (remove from unlocked_tasks)."""
        self.runtime.dependencies[task_id].add(name)
        self.runtime.unlocked_tasks.discard(task_id)

    def resolve_dependency(self, task_id: TaskID, name: str) -> None:
        """Remove a named dependency from a task; add to unlocked if all resolved."""
        dependencies = self.runtime.dependencies[task_id]

        dependencies.discard(name)
        if not dependencies:
            self.runtime.unlocked_tasks.add(task_id)

    def execute_task(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Commit a task to a machine and begin executing it at current time.

        Major DES operation: fixes the task assignment to the given machine,
        transitions the task to EXECUTING and queues TASK_STARTED and ASSIGNMENT
        events.

        Parameters
        ----------
        task_id : TaskID
            Task identifier.

        machine_id : MachineID
            Machine to execute on (must be a real machine, not GLOBAL_MACHINE_ID).

        Raises
        ------
        ValueError
            If machine_id == GLOBAL_MACHINE_ID.

        RuntimeError
            If task not available, machine infeasible, or presence prohibits execution.

        """
        if machine_id == GLOBAL_MACHINE_ID:
            raise ValueError(
                f"Cannot assign to the global machine {GLOBAL_MACHINE_ID}."
            )

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

        runtime.last_completion_time = max(
            runtime.last_completion_time, end_time
        )

        runtime.unlocked_tasks.remove(task_id)
        runtime.awaiting_tasks.remove(task_id)
        runtime.executing_tasks.add(task_id)
        runtime.status[task_id] = EXECUTING

        runtime.history[task_id].append(
            TaskHistory(machine_id, start_time, end_time)
        )

        if self._debug:
            validate_domain_bounds(
                task_id, self, machine_id=machine_id, origin="execute_task"
            )

    def pause_task(self, task_id: TaskID) -> None:
        """Suspend an executing task and allow rescheduling with reduced work.

        Major DES operation for preemption: computes work done proportionally to
        elapsed time, reduces remaining_time on all machines, and resets bounds to
        allow rescheduling from current time.
        Queues PAUSE and TASK_PAUSED events.

        Parameters
        ----------
        task_id : TaskID
            Executing task identifier.

        Raises
        ------
        RuntimeError
            If task is not currently executing.

        """
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

        feasible_machines = domains.feasible_machines[task_id]

        if expected_duration == 0:
            expected_duration = 1

        for m_id in self.instance.get_machines(task_id):
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

        if self._debug:
            validate_domain_bounds(task_id, self, origin="pause_task")

    # Runtime utils

    def get_next_start_lb(self) -> Time:
        """Return the minimum start lower bound among unlocked tasks, or current time if any task available now."""
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
        """Return the latest completion time seen so far in the schedule."""
        return self.runtime.last_completion_time

    def get_machine_execution(self) -> dict[MachineID, list[TaskID]]:
        """Return a dict mapping each machine to its currently executing task IDs."""
        assignments: dict[MachineID, list[TaskID]] = {
            machine_id: [] for machine_id in range(self.instance.n_machines)
        }

        runtime = self.runtime
        for task_id in runtime.executing_tasks:
            machine_id = runtime.get_assignment(task_id)
            assignments[machine_id].append(task_id)

        return assignments

    def __eq__(self, value: Any) -> bool:
        """Return equality based on all state attributes (instance, time, domains, runtime, events)."""
        return (
            isinstance(value, ScheduleState)
            and self.instance == value.instance
            and self.time == value.time
            and self.infeasible == value.infeasible
            and self._debug == value._debug
            and self.domains == value.domains
            and self.runtime == value.runtime
            and self.domain_event_queue == value.domain_event_queue
            and self.runtime_event_queue == value.runtime_event_queue
        )

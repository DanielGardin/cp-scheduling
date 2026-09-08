"""Scheduling Environment State Module.

This module provides ScheduleState, the core kernel for maintaining and querying
the state of a constraint satisfaction problem (CSP).
"""

from typing import Any

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    MIN_TIME,
    UNKNOWN_TASK,
    EzPickle,
    JobID,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state.csp import Presence, TaskDomains
from cpscheduler.environment.state.events import (
    DomainEventQueue,
    VarField,
)
from cpscheduler.environment.state.trail import Trail, TrailField
from cpscheduler.environment.utils.debug import (
    validate_domain_bounds,
    validate_machine_id,
)

PRESENT = Presence.PRESENT
ABSENT = Presence.ABSENT
INFEASIBLE = Presence.INFEASIBLE

ASSIGNMENT = VarField.ASSIGNMENT
START_LB = VarField.START_LB
START_UB = VarField.START_UB
END_LB = VarField.END_LB
END_UB = VarField.END_UB
PRESENCE = VarField.PRESENCE
ABSENCE = VarField.ABSENCE
MACHINE_INFEASIBLE = VarField.MACHINE_INFEASIBLE
STATE_INFEASIBLE = VarField.STATE_INFEASIBLE
GLOBAL_TIME = VarField.GLOBAL_TIME

T_START_LB = TrailField.START_LB
T_START_UB = TrailField.START_UB
T_END_LB = TrailField.END_LB
T_END_UB = TrailField.END_UB
T_FEASIBILITY = TrailField.FEASIBILITY
T_PRESENCE = TrailField.PRESENCE
T_FIXED = TrailField.FIXED
T_ASSIGNMENT = TrailField.ASSIGNMENT
T_REMAINING_TASKS = TrailField.REMAINING_TASKS
T_INFEASIBLE = TrailField.INFEASIBLE


# FUTURE: Study implementing backtracking functionality via trails
@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class ScheduleState(EzPickle):
    """Core state kernel for scheduling problems.

    ScheduleState maintains the constraint satisfaction problem (CSP) state
    (variable domains) for a scheduling environment.
    It provides a API to read and mutate state, delegating constraint
    propagation logic to the environment via event queues.

    """

    instance: ProblemInstance
    n_tasks: int
    n_jobs: int
    n_machines: int

    infeasible: bool
    remaining_tasks: int

    domains: TaskDomains

    domain_event_queue: DomainEventQueue
    trail: Trail

    _debug: bool

    def __init__(self, instance: ProblemInstance) -> None:
        """Initialize the ScheduleState with a problem instance.

        Parameters
        ----------
        instance: ProblemInstance
            The problem instance containing tasks, machines, processing times, etc.

        """
        self.instance = instance
        self.n_tasks = instance.n_tasks
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines

        self.infeasible = False
        self.remaining_tasks = self.n_tasks

        self._debug = instance.debug

        self.domains = TaskDomains(instance)

        self.domain_event_queue = DomainEventQueue()
        self.trail = Trail()

    # Properties
    @property
    def debug(self) -> bool:
        """Return whether debug mode is enabled for the state."""
        return self._debug

    # Flow control methods
    def reset(self) -> None:
        """Reset state to initial condition while preserving the problem instance."""
        self.infeasible = False
        self.remaining_tasks = self.instance.n_tasks

        self.domains = TaskDomains(self.instance)
        self.domain_event_queue.clear()
        self.trail.clear()

    def is_terminal(self) -> bool:
        """Return True if the problem is infeasible or all tasks are assigned."""
        return self.infeasible or self.remaining_tasks == 0

    def finish_propagation(self) -> None:
        """Flush the current event queue after a fixed-point iteration."""
        self.domain_event_queue.clear()

    # Problem Instance API methods

    ## Getter methods for instance parameters

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

    def get_job_id(self, task_id: TaskID) -> JobID:
        """Return the job the task belongs to."""
        return self.instance.job_ids[task_id]

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
        idx = task_id * self.n_machines + machine_id

        return self.domains.remaining_times[idx]

    def get_assignment(self, task_id: TaskID) -> MachineID:
        """Return the machine assigned to a task, or GLOBAL_MACHINE_ID if unassigned."""
        return self.domains.assignment[task_id]

    def get_machines(self, task_id: TaskID) -> list[MachineID]:
        """Return a list of currently feasible machines for a task."""
        machines = self.domains.machines
        start_idx, end_idx = machines.bounds(task_id)
        return machines.order[start_idx:end_idx]

    def is_fixed(self, task_id: TaskID) -> bool:
        """Return whether a task has been fixed."""
        return self.domains.fixed[task_id]

    def is_assigned(self, task_id: TaskID) -> bool:
        """Return whether a task has been assigned."""
        return self.domains.assignment[task_id] != GLOBAL_MACHINE_ID

    def is_locked(self, task_id: TaskID) -> bool:
        """Return whether a task has unresolved dependencies."""
        return bool(self.domains.dependencies[task_id])

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
        domains = self.domains

        if domains.presence[task_id] == INFEASIBLE:
            return False

        if machine_id == GLOBAL_MACHINE_ID:
            return bool(domains.machines.sizes[task_id])

        return domains.machines.is_feasible(task_id, machine_id)

    def can_start(
        self,
        task_id: TaskID,
        time: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> bool:
        """Return whether a task can be scheduled on the given machine at the time.

        A task is available if it is unlocked (all dependencies resolved) and the
        time falls within the feasible start window [start_lb, start_ub)
        on the machine(s).
        """
        domains = self.domains

        if (
            domains.dependencies[task_id]
            or domains.presence[task_id] == INFEASIBLE
        ):
            return False

        start = domains.start
        machines = domains.machines
        row = task_id * self.n_machines

        if machine_id == GLOBAL_MACHINE_ID:
            start_idx, end_idx = machines.bounds(task_id)
            order = machines.order

            for i in range(start_idx, end_idx):
                idx = row + order[i]
                if start.lbs[idx] <= time < start.ubs[idx]:
                    return True

            return False

        if not machines.is_feasible(task_id, machine_id):
            return False

        idx = row + machine_id
        return start.lbs[idx] <= time < start.ubs[idx]

    def get_unassigned_tasks(self) -> list[TaskID]:
        """Return a list of unassigned task IDs."""
        return [
            task_id
            for task_id, fixed in enumerate(self.domains.fixed)
            if not fixed
        ]

    def get_unlocked_tasks(self) -> list[TaskID]:
        """Return a list of unlocked task IDs (all dependencies resolved)."""
        dependencies = self.domains.dependencies

        return [
            task_id
            for task_id, fixed in enumerate(self.domains.fixed)
            if not fixed and not dependencies[task_id]
        ]

    def get_available_tasks(self, time: Time) -> list[TaskID]:
        """Return a list of available task IDs at that time."""
        return [
            task_id
            for task_id, fixed in enumerate(self.domains.fixed)
            if not fixed and self.can_start(task_id, time)
        ]

    def get_assigned_tasks(self) -> list[TaskID]:
        """Return a list of tasks with assigned machines."""
        presence = self.domains.presence

        return [
            task_id
            for task_id, fixed in enumerate(self.domains.fixed)
            if fixed and presence[task_id] == PRESENT
        ]

    ## Dependency-resolving methods
    def add_dependency(self, task_id: TaskID, name: str) -> None:
        """Add a named dependency to lock a task (remove from unlocked_tasks)."""
        deps = self.domains.dependencies[task_id]
        if name not in deps:
            if self.trail.active:
                self.trail.record_dependency(task_id, name, True)

            deps.add(name)

    def resolve_dependency(self, task_id: TaskID, name: str) -> None:
        """Remove a named dependency from a task; add to unlocked if all resolved."""
        deps = self.domains.dependencies[task_id]
        if name in deps:
            if self.trail.active:
                self.trail.record_dependency(task_id, name, False)

            deps.discard(name)

    ## Event-emitting methods
    def _restrict_presence(self, task_id: TaskID, mask: Presence) -> None:
        domains = self.domains
        trail = self.trail
        old_presence = domains.presence[task_id]
        new_presence = Presence(old_presence.value & mask.value)

        if new_presence == old_presence:
            return

        if new_presence == INFEASIBLE:
            if trail.active:
                trail.record(T_PRESENCE, old_presence.value, task_id)

            domains.presence[task_id] = INFEASIBLE
            self.fail(task_id)
            return

        if new_presence == PRESENT:
            field = PRESENCE

        elif new_presence == ABSENT:
            if trail.active:
                trail.record(T_FIXED, domains.fixed[task_id], task_id)
                trail.record(T_REMAINING_TASKS, self.remaining_tasks)

            domains.fixed[task_id] = True
            self.remaining_tasks -= 1
            field = ABSENCE

        else:
            raise RuntimeError(
                f"Unreachable: unexpected presence value {new_presence!r}"
            )

        if trail.active:
            trail.record(TrailField.PRESENCE, old_presence.value, task_id)

        domains.presence[task_id] = new_presence

        self.domain_event_queue.add_event(task_id, field)

    def require_task(self, task_id: TaskID) -> None:
        """Force a task to be present in the schedule."""
        self._restrict_presence(task_id, PRESENT)

    def forbid_task(self, task_id: TaskID) -> None:
        """Force a task to be absent in the schedule."""
        self._restrict_presence(task_id, ABSENT)

    def forbid_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Remove a machine from the feasible set for a task.

        If all machines are removed, marks the task as ABSENT.
        Otherwise, queues a MACHINE_INFEASIBLE event.
        """
        if machine_id == GLOBAL_MACHINE_ID:
            self.forbid_task(task_id)
            return

        domains = self.domains
        if not domains.machines.is_feasible(task_id, machine_id):
            return

        if self.trail.active:
            self.trail.record(
                T_FEASIBILITY, domains.machines.sizes[task_id], task_id
            )

        domains.machines.forbid(task_id, machine_id)

        if domains.machines.sizes[task_id]:
            domains.recompute_all_global_bounds(task_id)
            self.domain_event_queue.add_event(
                task_id, MACHINE_INFEASIBLE, machine_id
            )

        else:
            self._restrict_presence(task_id, ABSENT)

    def require_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Fix a task to run on a specific machine by forbidding all others."""
        if machine_id == GLOBAL_MACHINE_ID:
            return

        domains = self.domains

        if not domains.machines.is_feasible(task_id, machine_id):
            self.forbid_task(task_id)
            return

        machines = domains.machines
        order = machines.order
        start, end = machines.bounds(task_id)

        i = start
        while i < end:
            m_id = order[i]
            if m_id != machine_id:
                self.forbid_machine(task_id, m_id)
                start, end = machines.bounds(task_id)

            else:
                i += 1

    def tight_global_start_lb(self, task_id: TaskID, value: Time) -> None:
        """Raise the global lower bound of task start time (earliest start constraint)."""
        domains = self.domains
        if value <= domains.start.global_lbs[task_id]:
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        trail = self.trail
        machines = domains.machines
        order = machines.order
        row = task_id * self.n_machines

        start_idx, end_idx = machines.bounds(task_id)
        i = start_idx
        while i < end_idx:
            machine_id = order[i]
            idx = row + machine_id

            if value > start_lbs[idx]:
                end_lb = value + domains.remaining_times[idx]

                if trail.active:
                    trail.record(
                        T_START_LB, start_lbs[idx], task_id, machine_id
                    )
                    trail.record(T_END_LB, end_lbs[idx], task_id, machine_id)

                start_lbs[idx] = value
                end_lbs[idx] = end_lb

                if value > start_ubs[idx] or end_lb > end_ubs[idx]:
                    self.forbid_machine(task_id, machine_id)
                    start_idx, end_idx = machines.bounds(task_id)
                    continue

            i += 1

        domains.recompute_global_start_lbs(task_id)
        domains.recompute_global_end_lbs(task_id)

        if machines.sizes[task_id]:
            self.domain_event_queue.add_event(
                task_id=task_id,
                field=START_LB,
                time=value,
            )

    def tight_start_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID,
    ) -> None:
        """Raise the lower bound of task start time (earliest start constraint)."""
        domains = self.domains

        if value <= domains.start.get_lb(task_id, machine_id):
            return

        if not domains.machines.is_feasible(task_id, machine_id):
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        trail = self.trail
        idx = task_id * self.n_machines + machine_id

        old_lb = start_lbs[idx]
        end_lb = value + domains.remaining_times[idx]

        if trail.active:
            trail.record(T_START_LB, old_lb, task_id, machine_id)
            trail.record(T_END_LB, end_lbs[idx], task_id, machine_id)

        start_lbs[idx] = value
        end_lbs[idx] = end_lb

        if value > start_ubs[idx] or end_lb > end_ubs[idx]:
            self.forbid_machine(task_id, machine_id)
            return

        if old_lb == domains.start.global_lbs[task_id]:
            domains.recompute_global_start_lbs(task_id)
            domains.recompute_global_end_lbs(task_id)

        self.domain_event_queue.add_event(
            task_id=task_id,
            field=START_LB,
            machine_id=machine_id,
            time=value,
        )

    def tight_global_start_ub(self, task_id: TaskID, value: Time) -> None:
        """Lower the global upper bound of task start time (latest start constraint)."""
        domains = self.domains
        if value >= domains.start.global_ubs[task_id]:
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        trail = self.trail
        machines = domains.machines
        order = machines.order
        row = task_id * self.n_machines

        start_idx, end_idx = machines.bounds(task_id)
        i = start_idx
        while i < end_idx:
            machine_id = order[i]
            idx = row + machine_id

            if value < start_ubs[idx]:
                end_ub = value + domains.remaining_times[idx]

                if trail.active:
                    trail.record(
                        T_START_UB, start_ubs[idx], task_id, machine_id
                    )
                    trail.record(T_END_UB, end_ubs[idx], task_id, machine_id)

                start_ubs[idx] = value
                end_ubs[idx] = end_ub

                if value < start_lbs[idx] or end_ub < end_lbs[idx]:
                    self.forbid_machine(task_id, machine_id)
                    start_idx, end_idx = machines.bounds(task_id)
                    continue

            i += 1

        domains.recompute_global_start_ubs(task_id)
        domains.recompute_global_end_ubs(task_id)

        if machines.sizes[task_id]:
            self.domain_event_queue.add_event(
                task_id=task_id,
                field=START_UB,
                time=value,
            )

    def tight_start_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID,
    ) -> None:
        """Lower the upper bound of task start time (latest start constraint)."""
        domains = self.domains

        if value >= domains.start.get_ub(task_id, machine_id):
            return

        if not domains.machines.is_feasible(task_id, machine_id):
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        trail = self.trail
        idx = task_id * self.n_machines + machine_id

        old_ub = start_ubs[idx]
        end_ub = value + domains.remaining_times[idx]

        if trail.active:
            trail.record(T_START_UB, old_ub, task_id, machine_id)
            trail.record(T_END_UB, end_ubs[idx], task_id, machine_id)

        start_ubs[idx] = value
        end_ubs[idx] = end_ub

        if value < start_lbs[idx] or end_ub < end_lbs[idx]:
            self.forbid_machine(task_id, machine_id)
            return

        if old_ub == domains.start.global_ubs[task_id]:
            domains.recompute_global_start_ubs(task_id)
            domains.recompute_global_end_ubs(task_id)

        self.domain_event_queue.add_event(
            task_id=task_id,
            field=START_UB,
            machine_id=machine_id,
            time=value,
        )

    def tight_global_end_lb(self, task_id: TaskID, value: Time) -> None:
        """Raise the global lower bound of task end time (earliest end constraint)."""
        domains = self.domains
        if value <= domains.end.global_lbs[task_id]:
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        trail = self.trail
        machines = domains.machines
        order = machines.order
        row = task_id * self.n_machines

        start_idx, end_idx = machines.bounds(task_id)
        i = start_idx
        while i < end_idx:
            machine_id = order[i]
            idx = row + machine_id

            if value > end_lbs[idx]:
                start_lb = value - domains.remaining_times[idx]

                if trail.active:
                    trail.record(T_END_LB, end_lbs[idx], task_id, machine_id)
                    trail.record(
                        T_START_LB, start_lbs[idx], task_id, machine_id
                    )

                end_lbs[idx] = value
                start_lbs[idx] = start_lb

                if start_lb > start_ubs[idx] or value > end_ubs[idx]:
                    self.forbid_machine(task_id, machine_id)
                    start_idx, end_idx = machines.bounds(task_id)
                    continue

            i += 1

        domains.recompute_global_start_lbs(task_id)
        domains.recompute_global_end_lbs(task_id)

        if machines.sizes[task_id]:
            self.domain_event_queue.add_event(
                task_id=task_id,
                field=END_LB,
                time=value,
            )

    def tight_end_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID,
    ) -> None:
        """Raise the lower bound of task end time (earliest end constraint)."""
        domains = self.domains

        if value <= domains.end.get_lb(task_id, machine_id):
            return

        if not domains.machines.is_feasible(task_id, machine_id):
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        trail = self.trail
        idx = task_id * self.n_machines + machine_id

        old_lb = end_lbs[idx]
        start_lb = value - domains.remaining_times[idx]

        if trail.active:
            trail.record(T_END_LB, old_lb, task_id, machine_id)
            trail.record(T_START_LB, start_lbs[idx], task_id, machine_id)

        end_lbs[idx] = value
        start_lbs[idx] = start_lb

        if start_lb > start_ubs[idx] or value > end_ubs[idx]:
            self.forbid_machine(task_id, machine_id)
            return

        if old_lb == domains.end.global_lbs[task_id]:
            domains.recompute_global_start_lbs(task_id)
            domains.recompute_global_end_lbs(task_id)

        self.domain_event_queue.add_event(
            task_id=task_id,
            field=END_LB,
            machine_id=machine_id,
            time=value,
        )

    def tight_global_end_ub(self, task_id: TaskID, value: Time) -> None:
        """Lower the global upper bound of task end time (latest end constraint)."""
        domains = self.domains
        if value >= domains.end.global_ubs[task_id]:
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        trail = self.trail
        machines = domains.machines
        order = machines.order
        row = task_id * self.n_machines

        start_idx, end_idx = machines.bounds(task_id)
        i = start_idx
        while i < end_idx:
            machine_id = order[i]
            idx = row + machine_id

            if value < end_ubs[idx]:
                start_ub = value - domains.remaining_times[idx]

                if trail.active:
                    trail.record(T_END_UB, end_ubs[idx], task_id, machine_id)
                    trail.record(
                        T_START_UB, start_ubs[idx], task_id, machine_id
                    )

                end_ubs[idx] = value
                start_ubs[idx] = start_ub

                if value < end_lbs[idx] or start_ub < start_lbs[idx]:
                    self.forbid_machine(task_id, machine_id)
                    start_idx, end_idx = machines.bounds(task_id)
                    continue

            i += 1

        domains.recompute_global_start_ubs(task_id)
        domains.recompute_global_end_ubs(task_id)

        if machines.sizes[task_id]:
            self.domain_event_queue.add_event(
                task_id=task_id,
                field=END_UB,
                time=value,
            )

    def tight_end_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID,
    ) -> None:
        """Lower the upper bound of task end time (latest end constraint)."""
        domains = self.domains

        if value >= domains.end.get_ub(task_id, machine_id):
            return

        if not domains.machines.is_feasible(task_id, machine_id):
            return

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs
        trail = self.trail
        idx = task_id * self.n_machines + machine_id

        old_ub = end_ubs[idx]
        start_ub = value - domains.remaining_times[idx]

        if trail.active:
            trail.record(T_END_UB, old_ub, task_id, machine_id)
            trail.record(T_START_UB, start_ubs[idx], task_id, machine_id)

        end_ubs[idx] = value
        start_ubs[idx] = start_ub

        if value < end_lbs[idx] or start_ub < start_lbs[idx]:
            self.forbid_machine(task_id, machine_id)
            return

        if old_ub == domains.end.global_ubs[task_id]:
            domains.recompute_global_start_ubs(task_id)
            domains.recompute_global_end_ubs(task_id)

        self.domain_event_queue.add_event(
            task_id=task_id,
            field=END_UB,
            machine_id=machine_id,
            time=value,
        )

    def assign_task(
        self,
        task_id: TaskID,
        machine_id: MachineID,
        start_time: Time,
    ) -> None:
        """Commit a task to a machine and begin executing it at current time.

        Fixes the task assignment to the given machine, queues ASSIGNMENT events.

        Parameters
        ----------
        task_id : TaskID
            Task identifier.

        machine_id : MachineID
            Machine to execute on (must be a real machine, not GLOBAL_MACHINE_ID).

        start_time: Time
            Time to execute the task, must be a feasible start time.

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

        domains = self.domains
        if domains.assignment[task_id] != GLOBAL_MACHINE_ID:
            raise RuntimeError(
                f"Task {task_id} have already been assigned before."
            )

        if self.debug:
            validate_machine_id(
                task_id,
                machine_id,
                self.instance,
                origin="assign_task",
                allow_global=False,
            )

            if not domains.machines.is_feasible(task_id, machine_id):
                lb = domains.start.get_lb(task_id, machine_id)
                ub = domains.start.get_ub(task_id, machine_id)

                raise RuntimeError(
                    f"Cannot assign task {task_id} to machine {machine_id} at time "
                    f"{start_time}, because this machine is not feasible. "
                    f"Start Interval = [{lb}, {ub}]."
                )

            presence = domains.presence[task_id]
            if not presence.contains_present():
                raise RuntimeError(
                    f"Cannot assign task {task_id} to machine {machine_id} at time "
                    f"{start_time}, it violates the presence constraints for that "
                    f"task: presence = {presence.name}."
                )

        self.require_task(task_id)
        self.require_machine(task_id, machine_id)
        self.tight_start_lb(task_id, start_time, machine_id)
        self.tight_start_ub(task_id, start_time, machine_id)

        if self.infeasible:
            return

        trail = self.trail
        if trail.active:
            trail.record(T_FIXED, False, task_id)
            trail.record(T_ASSIGNMENT, GLOBAL_MACHINE_ID, task_id)
            trail.record(T_REMAINING_TASKS, self.remaining_tasks)

        domains.fixed[task_id] = True
        domains.assignment[task_id] = machine_id
        self.remaining_tasks -= 1

        self.domain_event_queue.add_event(
            task_id=task_id,
            field=ASSIGNMENT,
            machine_id=machine_id,
            time=start_time,
        )

        if self._debug:
            validate_domain_bounds(
                task_id, self, machine_id=machine_id, origin="assign_task"
            )

    def tight_global_time(self, time: Time) -> None:
        """Contraint all tasks to only execute after a given time."""
        self.domain_event_queue.add_event(
            task_id=UNKNOWN_TASK,
            field=GLOBAL_TIME,
            time=time,
        )

    def fail(self, task_id: TaskID = UNKNOWN_TASK) -> None:
        """Mark the problem as infeasible.

        Constraints should prefer domain reductions via forbid_machine or
        tight_* methods instead of this method.
        It is reserved for hard global conflicts or defensive safeguards.
        """
        if self.infeasible:
            return

        if self.trail.active:
            self.trail.record(T_INFEASIBLE, False)

        self.infeasible = True
        self.domain_event_queue.add_event(task_id, STATE_INFEASIBLE)

    # Backtrack search API

    def checkpoint(self) -> int:
        """Push a backtracking checkpoint."""
        if self.domain_event_queue:
            raise RuntimeError("Cannot create a checkpoint mid propagation.")

        return self.trail.mark()

    def backtrack(self, mark: int) -> set[TaskID]:
        """Undo all mutations recorded after checkpoint mark."""
        if self.domain_event_queue:
            raise RuntimeError(
                "Cannot backtrack while propagation events are pending."
            )

        trail = self.trail

        if not trail.has_mark(mark):
            raise RuntimeError(
                f"No checkpoint mark {mark} exists, "
                f"there are only {trail.n_marks} checkpoints recorded."
            )

        field_mark = trail.marks[mark]
        dep_mark = trail.dep_marks[mark]

        domains = self.domains
        changed_tasks: set[TaskID] = set()

        dep_log = trail.dep_log
        for i in range(len(dep_log) - 1, dep_mark - 1, -1):
            dep_task_id, name, added = dep_log[i]
            deps = domains.dependencies[dep_task_id]

            if added:
                deps.discard(name)

            else:
                deps.add(name)

        del dep_log[dep_mark:]

        fields = trail.fields
        tasks = trail.tasks
        machines = trail.machines
        values = trail.values
        n_machines = self.n_machines

        for i in range(len(fields) - 1, field_mark - 1, -1):
            field = fields[i]
            task_id = tasks[i]
            machine_id = machines[i]
            old_value = values[i]

            if field == T_START_LB:
                domains.start.lbs[task_id * n_machines + machine_id] = old_value
                changed_tasks.add(task_id)

            elif field == T_START_UB:
                domains.start.ubs[task_id * n_machines + machine_id] = old_value
                changed_tasks.add(task_id)

            elif field == T_END_LB:
                domains.end.lbs[task_id * n_machines + machine_id] = old_value
                changed_tasks.add(task_id)

            elif field == T_END_UB:
                domains.end.ubs[task_id * n_machines + machine_id] = old_value
                changed_tasks.add(task_id)

            elif field == T_FEASIBILITY:
                domains.machines.restore_size(task_id, old_value)
                changed_tasks.add(task_id)

            elif field == T_PRESENCE:
                domains.presence[task_id] = Presence(old_value)
                changed_tasks.add(task_id)

            elif field == T_FIXED:
                domains.fixed[task_id] = bool(old_value)
                changed_tasks.add(task_id)

            elif field == T_ASSIGNMENT:
                domains.assignment[task_id] = old_value
                changed_tasks.add(task_id)

            elif field == T_REMAINING_TASKS:
                self.remaining_tasks = old_value

            elif field == T_INFEASIBLE:
                self.infeasible = bool(old_value)

        del trail.marks[mark:]
        del trail.dep_marks[mark:]
        del fields[field_mark:]
        del tasks[field_mark:]
        del machines[field_mark:]
        del values[field_mark:]

        for task_id in changed_tasks:
            domains.restore_task(task_id)

        if not trail.marks:
            trail.active = False

        return changed_tasks

    # Runtime utils

    def get_start(self, task_id: TaskID) -> Time:
        """Return the start time of a fixed task."""
        if not self.domains.fixed[task_id]:
            raise ValueError(f"Task {task_id} is not fixed yet.")

        if self.domains.presence[task_id] != PRESENT:
            raise ValueError(f"Task {task_id} is not present.")

        return self.domains.start.get_global_ub(task_id)

    def get_end(self, task_id: TaskID) -> Time:
        """Return the end time of a fixed task."""
        if not self.domains.fixed[task_id]:
            raise ValueError(f"Task {task_id} is not fixed yet.")

        if self.domains.presence[task_id] != PRESENT:
            raise ValueError(f"Task {task_id} is not present.")

        return self.domains.end.get_global_ub(task_id)

    def get_earliest_start_lb(self) -> Time:
        """Return the earliest start lower bound among unlocked tasks."""
        global_lbs = self.domains.start.global_lbs
        dependencies = self.domains.dependencies

        min_lb = MAX_TIME
        for task_id, fixed in enumerate(self.domains.fixed):
            if fixed or dependencies[task_id]:
                continue

            lb = global_lbs[task_id]

            if lb < min_lb:
                min_lb = lb

        return min_lb

    def get_next_decision_point(self, time: Time) -> Time:
        """Return the earliest start lower bound greater than `time`."""
        global_lbs = self.domains.start.global_lbs
        dependencies = self.domains.dependencies

        next_time = MAX_TIME
        for task_id, fixed in enumerate(self.domains.fixed):
            if fixed or dependencies[task_id]:
                continue

            lb = global_lbs[task_id]

            if time < lb < next_time:
                next_time = lb

        return next_time

    def get_latest_end(self) -> Time:
        """Return the end time of the latest task."""
        ends = self.domains.end.global_lbs

        max_end = MIN_TIME
        for task_id, fixed in enumerate(self.domains.fixed):
            if not fixed:
                continue

            end = ends[task_id]

            if end > max_end:
                max_end = end

        return max_end

    def __eq__(self, value: Any) -> bool:
        """Return equality based on all state attributes (instance, time, domains, runtime, events)."""
        return (
            isinstance(value, ScheduleState)
            and self.instance == value.instance
            and self.domains == value.domains
            and self.domain_event_queue == value.domain_event_queue
            and self._debug == value._debug
        )

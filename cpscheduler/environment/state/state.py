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

UNKNOWN_TASK: TaskID = -1


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

    def is_terminal(self) -> bool:
        """Return True if the problem is infeasible or all tasks are assigned."""
        return self.infeasible or self.remaining_tasks == 0

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

    def get_machines(self, task_id: TaskID) -> tuple[MachineID, ...]:
        """Return the tuple of currently feasible machines for a task."""
        return self.domains.get_feasible_machines(task_id)

    def is_fixed(self, task_id: TaskID) -> bool:
        """Return whether a task has been fixed."""
        return self.domains.fixed[task_id]

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
        if self.domains.presence[task_id] == INFEASIBLE:
            return False

        machines = self.domains.feasible_machines[task_id]

        if machine_id == GLOBAL_MACHINE_ID:
            return bool(machines)

        return machine_id in machines

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
        if self.domains.dependencies[task_id]:
            return False

        start = self.domains.start

        if machine_id == GLOBAL_MACHINE_ID:
            row = task_id * self.n_machines

            return any(
                start.lbs[row + m_id] <= time < start.ubs[row + m_id]
                for m_id in self.domains.feasible_machines[task_id]
            )

        if machine_id not in self.domains.feasible_machines[task_id]:
            return False

        idx = task_id * self.n_machines + machine_id
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
        self.domains.dependencies[task_id].add(name)

    def resolve_dependency(self, task_id: TaskID, name: str) -> None:
        """Remove a named dependency from a task; add to unlocked if all resolved."""
        self.domains.dependencies[task_id].discard(name)

    ## Event-emitting methods
    def _restrict_presence(self, task_id: TaskID, mask: Presence) -> None:
        domains = self.domains
        old_presence = domains.presence[task_id]
        # Bitwise operations on Literal unions are inferred as int by type checkers.
        # Explicitly narrow back to PresenceType.
        new_presence = Presence(old_presence.value & mask.value)

        if new_presence == old_presence:
            return

        if new_presence == INFEASIBLE:
            domains.presence[task_id] = INFEASIBLE
            self.fail(task_id)
            return

        if new_presence == PRESENT:
            field = PRESENCE

        elif new_presence == ABSENT:
            domains.fixed[task_id] = True
            self.remaining_tasks -= 1
            field = ABSENCE

        else:
            raise RuntimeError(
                f"Unreachable: unexpected presence value {new_presence!r}"
            )

        domains.presence[task_id] = new_presence

        self.domain_event_queue.add_event(task_id, field)

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
            domains.recompute_all_global_bounds(task_id)

            self.domain_event_queue.add_event(
                task_id, MACHINE_INFEASIBLE, machine_id
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

        if machine_id != GLOBAL_MACHINE_ID:
            idx = task_id * self.n_machines + machine_id

            old_lb = start_lbs[idx]
            end_lb = value + domains.remaining_times[idx]

            start_lbs[idx] = value
            end_lbs[idx] = end_lb

            if value > start_ubs[idx] or end_lb > end_ubs[idx]:
                self.restrict_machine(task_id, machine_id)
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
            return

        row = task_id * self.n_machines
        for machine_id in domains.get_feasible_machines(task_id):
            idx = row + machine_id

            if value > start_lbs[idx]:
                end_lb = value + domains.remaining_times[idx]

                start_lbs[idx] = value
                end_lbs[idx] = end_lb

                if value > start_ubs[idx] or end_lb > end_ubs[idx]:
                    self.restrict_machine(task_id, machine_id)

        domains.recompute_global_start_lbs(task_id)
        domains.recompute_global_end_lbs(task_id)

        if domains.feasible_machines[task_id]:
            self.domain_event_queue.add_event(
                task_id=task_id,
                field=START_LB,
                time=value,
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
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs

        if machine_id != GLOBAL_MACHINE_ID:
            idx = task_id * self.n_machines + machine_id

            old_ub = start_ubs[idx]
            end_ub = value + domains.remaining_times[idx]

            start_ubs[idx] = value
            end_ubs[idx] = end_ub

            if start_lbs[idx] > value or end_lbs[idx] > end_ub:
                self.restrict_machine(task_id, machine_id)
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
            return

        row = task_id * self.n_machines

        for machine_id in domains.get_feasible_machines(task_id):
            idx = row + machine_id

            if value < start_ubs[idx]:
                end_ub = value + domains.remaining_times[idx]

                start_ubs[idx] = value
                end_ubs[idx] = end_ub

                if start_lbs[idx] > value or end_lbs[idx] > end_ub:
                    self.restrict_machine(task_id, machine_id)

        domains.recompute_global_start_ubs(task_id)
        domains.recompute_global_end_ubs(task_id)

        if domains.feasible_machines[task_id]:
            self.domain_event_queue.add_event(
                task_id=task_id,
                field=START_UB,
                time=value,
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

        if machine_id != GLOBAL_MACHINE_ID:
            idx = task_id * self.n_machines + machine_id

            old_lb = end_lbs[idx]
            start_lb = value - domains.remaining_times[idx]

            end_lbs[idx] = value
            start_lbs[idx] = start_lb

            if start_lb > start_ubs[idx] or value > end_ubs[idx]:
                self.restrict_machine(task_id, machine_id)
                return

            if old_lb == domains.end.global_lbs[task_id]:
                domains.recompute_global_end_lbs(task_id)
                domains.recompute_global_start_lbs(task_id)

            self.domain_event_queue.add_event(
                task_id=task_id,
                field=END_LB,
                machine_id=machine_id,
                time=value,
            )
            return

        row = task_id * self.n_machines

        for machine_id in domains.get_feasible_machines(task_id):
            idx = row + machine_id

            if value > end_lbs[idx]:
                start_lb = value - domains.remaining_times[idx]

                end_lbs[idx] = value
                start_lbs[idx] = start_lb

                if start_lb > start_ubs[idx] or value > end_ubs[idx]:
                    self.restrict_machine(task_id, machine_id)

        domains.recompute_global_end_lbs(task_id)
        domains.recompute_global_start_lbs(task_id)

        if domains.feasible_machines[task_id]:
            self.domain_event_queue.add_event(
                task_id=task_id,
                field=END_LB,
                time=value,
            )

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

        start_lbs = domains.start.lbs
        start_ubs = domains.start.ubs
        end_lbs = domains.end.lbs
        end_ubs = domains.end.ubs

        if machine_id != GLOBAL_MACHINE_ID:
            idx = task_id * self.n_machines + machine_id

            old_ub = end_ubs[idx]
            start_ub = value - domains.remaining_times[idx]

            end_ubs[idx] = value
            start_ubs[idx] = start_ub

            if start_lbs[idx] > start_ub or end_lbs[idx] > value:
                self.restrict_machine(task_id, machine_id)
                return

            if old_ub == domains.end.global_ubs[task_id]:
                domains.recompute_global_end_ubs(task_id)
                domains.recompute_global_start_ubs(task_id)

            self.domain_event_queue.add_event(
                task_id=task_id,
                field=END_UB,
                machine_id=machine_id,
                time=value,
            )

            return

        row = task_id * self.n_machines

        for machine_id in domains.get_feasible_machines(task_id):
            idx = row + machine_id

            if end_ubs[idx] > value:
                start_ub = value - domains.remaining_times[idx]

                end_ubs[idx] = value
                start_ubs[idx] = start_ub

                if start_lbs[idx] > start_ub or end_lbs[idx] > value:
                    self.restrict_machine(task_id, machine_id)

        domains.recompute_global_end_ubs(task_id)
        domains.recompute_global_start_ubs(task_id)

        if domains.feasible_machines[task_id]:
            self.domain_event_queue.add_event(
                task_id=task_id,
                field=END_UB,
                time=value,
            )

    def forbid_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Remove a machine from the feasible set of a task."""
        self.restrict_machine(task_id, machine_id)

    def require_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        """Fix a task to run on a specific machine by forbidding all others."""
        for other_machine in self.domains.feasible_machines[task_id]:
            if other_machine != machine_id:
                self.forbid_machine(task_id, other_machine)

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

        if self.debug:
            validate_machine_id(
                task_id,
                machine_id,
                self.instance,
                origin="assign_task",
                allow_global=False,
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
            if not presence.contains_present():
                raise RuntimeError(
                    f"Cannot assign task {task_id} to machine {machine_id} at time "
                    f"{start_time}, it violates the presence constraints for that "
                    f"task: presence = {presence.name}."
                )

        self.require_task(task_id)
        self.tight_start_lb(task_id, start_time, machine_id)
        self.tight_start_ub(task_id, start_time, machine_id)

        if self.infeasible:
            return

        domains.assign(task_id, machine_id)
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

        Constraints should prefer domain reductions via restrict_machine or
        tight_* methods instead of this method.
        It is reserved for hard global conflicts or defensive safeguards.
        """
        self.infeasible = True
        self.domain_event_queue.add_event(task_id, STATE_INFEASIBLE)

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

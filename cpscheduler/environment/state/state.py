from typing import Any, TypeAlias

from cpscheduler.environment.constants import (
    MAX_TIME,
    MachineID,
    TaskID,
    Time,
    GLOBAL_MACHINE_ID,
)
from cpscheduler.environment.constants import Status

from cpscheduler.environment.state.csp import ScheduleVariables, Presence
from cpscheduler.environment.state.events import DomainEvent
from cpscheduler.environment.state.instance import ProblemInstance
from cpscheduler.environment.state.runtime import RuntimeState, RuntimeEvent

ObsType: TypeAlias = tuple[dict[str, list[Any]], dict[str, list[Any]]]

PRESENT = Presence.PRESENT
ABSENT = Presence.ABSENT
INFEASIBLE = Presence.INFEASIBLE

AWAITING = Status.AWAITING
PAUSED = Status.PAUSED
EXECUTING = Status.EXECUTING
COMPLETED = Status.COMPLETED

class ScheduleState:
    """
    ScheduleState represents the current state of the scheduling environment,
    working as both a Discrete Event Simulation (DES) state and a Constraint
    Satisfaction Problem (CSP) state.

    It has no simulation logic itself, instead, it provides an API to read and
    modify the current state of the problem. The actual kernel is implemented in
    the SchedulingEnv class, which knows the constraints and how to propagate
    changes through them.
    """

    __slots__ = (
        "instance",
        "time",
        "_variables",
        "runtime_state",
        "loaded",
    )

    instance: ProblemInstance

    time: Time
    "Current simulation time."

    _variables: ScheduleVariables
    "Encapsulates all the variable data for the tasks"

    runtime_state: RuntimeState
    "Encapsulates the runtime state of the tasks, such as their history and current status"

    loaded: bool

    def __init__(self) -> None:
        self.time = 0
        self.loaded = False

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

    # Instance control methods
    def read_instance(
        self,
        task_data: dict[str, list[Any]],
    ) -> None:
        self.instance = ProblemInstance(task_data)
        self.loaded = True

    # Flow control methods
    def reset(self) -> None:
        assert (
            self.loaded
        ), "No instance loaded. Please load an instance before resetting the state."

        self.time = 0

        self._variables = ScheduleVariables(self.instance)
        self.runtime_state = RuntimeState(self.instance)

    def get_event_queue(self) -> list[DomainEvent]:
        return self._variables.event_queue

    def clear_event_queue(self) -> None:
        self._variables.event_queue.clear()

    def get_runtime_event_queue(self) -> list[RuntimeEvent]:
        return self.runtime_state.event_queue

    def clear_runtime_event_queue(self) -> None:
        self.runtime_state.event_queue.clear()

    def is_terminal(self) -> bool:
        return self._variables.infeasible or (
            not self._variables.awaiting_tasks and
            not self.runtime_state.executing_tasks
        )

    def advance_time_(self, new_time: Time) -> None:
        assert (
            new_time > self.time
        ), "Advance time must be monotonic increasing."

        self.time = new_time
        self.runtime_state.update(self.time)

    # Constraint propagation API methods

    ## Getter methods for variable values
    def get_start_lb(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self._variables.start.get_lb(task_id, machine_id)

    def get_start_ub(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self._variables.start.get_ub(task_id, machine_id)

    def get_end_lb(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self._variables.end.get_lb(task_id, machine_id)

    def get_end_ub(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self._variables.end.get_ub(task_id, machine_id)

    def get_remaining_time(
        self, task_id: TaskID, machine_id: MachineID
    ) -> Time:
        return self._variables.remaining_times[
            task_id * self.instance.n_machines + machine_id
        ]

    def get_assignment(self, task_id: TaskID) -> MachineID:
        return self._variables.assignment[task_id]

    def get_machines(self, task_id: TaskID) -> list[MachineID]:
        return self._variables.feasible_machines[task_id]

    def is_fixed(self, task_id: TaskID) -> bool:
        return self._variables.fixed[task_id]

    def is_present(self, task_id: TaskID) -> bool:
        return self._variables.presence[task_id] == PRESENT

    def is_absent(self, task_id: TaskID) -> bool:
        return self._variables.presence[task_id] == ABSENT

    def is_feasible(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:

        if self._variables.presence[task_id] == INFEASIBLE:
            return False

        machines = self._variables.feasible_machines[task_id]

        if machine_id == GLOBAL_MACHINE_ID:
            return bool(machines)

        return machine_id in machines

    ## Setter methods for variable values, triggering constraint propagation through events
    def require_task(self, task_id: TaskID) -> None:
        self._variables.require_task(task_id)

    def forbid_task(self, task_id: TaskID) -> None:
        self._variables.forbid_task(task_id)

    def tight_start_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        variables = self._variables

        if value <= variables.start.get_lb(task_id, machine_id):
            return

        if machine_id == GLOBAL_MACHINE_ID:
            variables.set_start_lb(task_id, value)

        else:
            variables.set_machine_start_lb(task_id, value, machine_id)

    def tight_start_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        variables = self._variables

        if value >= variables.start.get_ub(task_id, machine_id):
            return

        if machine_id == GLOBAL_MACHINE_ID:
            variables.set_start_ub(task_id, value)

        else:
            variables.set_machine_start_ub(task_id, value, machine_id)

    def tight_end_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        variables = self._variables

        if value <= variables.end.get_lb(task_id, machine_id):
            return

        if machine_id == GLOBAL_MACHINE_ID:
            variables.set_end_lb(task_id, value)

        else:
            variables.set_machine_end_lb(task_id, value, machine_id)

    def tight_end_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        variables = self._variables

        if value >= variables.end.get_ub(task_id, machine_id):
            return

        if machine_id == GLOBAL_MACHINE_ID:
            variables.set_end_ub(task_id, value)

        else:
            variables.set_machine_end_ub(task_id, value, machine_id)

    def forbid_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        self._variables.restrict_machine(task_id, machine_id)

    def require_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        for other_machine in self.instance.processing_times[task_id]:
            if other_machine != machine_id:
                self.forbid_machine(task_id, other_machine)

    def reset_bounds(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> None:
        self._variables.reset_bounds(task_id, self.time, machine_id)

    def fail(self) -> None:
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
        self._variables.set_infeasible_state()

    # Discrete event simulation API methods

    ## Getter methods for variable values
    def is_awaiting(self, task_id: TaskID) -> bool:
        return self.runtime_state.status[task_id] == AWAITING

    def is_paused(self, task_id: TaskID) -> bool:
        return self.runtime_state.status[task_id] == PAUSED

    def is_executing(self, task_id: TaskID) -> bool:
        return self.runtime_state.status[task_id] == EXECUTING

    def is_completed(self, task_id: TaskID) -> bool:
        return self.runtime_state.status[task_id] == COMPLETED

    def is_available(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:
        return (
            self.is_awaiting(task_id) and
            self._is_available(task_id, machine_id)
        )

    def _is_available(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:
        start_var = self._variables.start
        t = self.time

        lb = start_var.get_lb(task_id, machine_id)
        if t < lb:
            return False

        ub = start_var.get_ub(task_id, machine_id)

        return t < ub

    def get_end(self, task_id: TaskID) -> Time:
        if not self.is_fixed(task_id):
            raise RuntimeError(f"Task {task_id} has not been commited yet.")

        return self.runtime_state.get_end(task_id)
    
    def get_start(self, task_id: TaskID) -> Time:
        if not self.is_fixed(task_id):
            raise RuntimeError(f"Task {task_id} has not been commited yet.")

        return self.runtime_state.get_start(task_id)

    ## Setter methods for variable values, triggering constraint propagation through events
    def execute_task(self, task_id: TaskID, machine_id: MachineID) -> None:
        start_time = self.time

        self._variables.assign(task_id, start_time, machine_id)

        end_time = start_time + self.get_remaining_time(task_id, machine_id)
        self.runtime_state.start_task(task_id, machine_id, start_time, end_time)

    def pause_task(self, task_id: TaskID) -> None:
        self._variables.pause(task_id, self.time)

        self.runtime_state.pause_task(task_id, self.time)

    # Runtime utils

    #$ :---:
    # These have a weird aftertaste: They are often used for iterating over
    # them, but the set can change between elements. This makes a situation,
    # where a posterior element is not in the set anymore, due to a change
    # triggered by a previous element, possible.
    def get_awaiting_tasks(self) -> list[TaskID]:
        return list(self._variables.awaiting_tasks)
    
    def get_executing_tasks(self) -> list[TaskID]:
        return list(self.runtime_state.executing_tasks)

    def get_completed_tasks(self) -> list[TaskID]:
        return list(self.runtime_state.completed_tasks)
    # A solution would be having an generator that builds the list, and check
    # ownership for each element before yielding it, but I think that is too
    # much of a worry for a small issue that may be handled by CSP guardrails.
    #$ :---:

    def get_next_start_lb(self) -> Time:
        min_lb = MAX_TIME

        global_lbs = self._variables.start.global_lbs
        awaiting_tasks = self._variables.awaiting_tasks
        current_time = self.time

        for task_id in awaiting_tasks:
            lb = global_lbs[task_id]

            if lb <= current_time:
                return current_time

            if lb < min_lb:
                min_lb = global_lbs[task_id]

        return min_lb

    def get_last_completion_time(self) -> Time:
        return self.runtime_state.last_completion_time

    def get_machine_execution(self) -> dict[MachineID, list[TaskID]]:
        assignments: dict[MachineID, list[TaskID]] = {
            machine_id: [] for machine_id in range(self.instance.n_machines)
        }

        for task_id in self.runtime_state.executing_tasks:
            machine_id = self.runtime_state.get_assignment(task_id)
            assignments[machine_id].append(task_id)

        return assignments

    def get_observation(self) -> ObsType:
        task_obs = self.instance.task_instance.copy()
        available = [False] * self.n_tasks

        task_obs["status"] = self.runtime_state.status.copy()
        for task_id in self._variables.awaiting_tasks:
            available[task_id] = self._is_available(task_id)

        task_obs["available"] = available

        return task_obs, {}

    def __reduce__(self) -> str | tuple[Any, ...]:
        state = (
            self.instance,
            self.time,
            self._variables,
            self.runtime_state,
            self.loaded,
        )
        return (self.__class__, (), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.instance,
            self.time,
            self._variables,
            self.runtime_state,
            self.loaded,
        ) = state

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduleState):
            return NotImplemented

        return (
            self.instance == other.instance
            and self._variables == other._variables
            and self.runtime_state == other.runtime_state
            and self.time == other.time
        )

    def __repr__(self) -> str:
        return (
            f"ScheduleState(time={self.time}, "
            f"instance={self.instance}, "
            f"variables={self._variables}, "
            f"runtime_state={self.runtime_state})"
        )

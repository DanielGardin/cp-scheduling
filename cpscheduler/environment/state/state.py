from typing import Any
from typing_extensions import TypeAlias

from cpscheduler.environment.constants import (
    MAX_TIME,
    MachineID,
    TaskID,
    Time,
    GLOBAL_MACHINE_ID
)
from cpscheduler.environment.state.events import Event
from cpscheduler.environment.state.csp import ScheduleVariables, PRESENT, ABSENT
from cpscheduler.environment.state.des import ProblemInstance
from cpscheduler.environment.state.runtime import RuntimeState

ObsType: TypeAlias = tuple[dict[str, list[Any]], dict[str, list[Any]]]


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
        "variables_",
        "runtime_state",
        "event_queue",
        "infeasible",
        "loaded",
    )

    instance: ProblemInstance

    time: Time
    "Current simulation time."

    variables_: ScheduleVariables
    "Encapsulates all the variable data for the tasks"

    runtime_state: RuntimeState
    "Encapsulates the runtime state of the tasks, such as their history and current status"

    event_queue: list[Event]
    "Queue of events to be processed for constraint propagation."

    loaded: bool
    infeasible: bool

    def __init__(self) -> None:
        self.time = 0
        self.event_queue = []

        self.infeasible = False
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

        self.event_queue.clear()
        self.variables_ = ScheduleVariables(self.instance)
        self.runtime_state = RuntimeState(self.instance)

    def is_terminal(self) -> bool:
        return self.infeasible or self.runtime_state.is_terminal()

    def advance_time(self, new_time: Time) -> None:
        assert new_time >= self.time, "Cannot go back in time."

        self.time = new_time
        self.runtime_state.update(self.time)

    # Constraint propagation API methods

    ## Getter methods for variable values
    def get_start_lb(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self.variables_.start.get_lb(task_id, machine_id)

    def get_start_ub(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self.variables_.start.get_ub(task_id, machine_id)

    def get_end_lb(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self.variables_.end.get_lb(task_id, machine_id)

    def get_end_ub(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> Time:
        return self.variables_.end.get_ub(task_id, machine_id)

    def get_remaining_time(
        self, task_id: TaskID, machine_id: MachineID
    ) -> Time:
        return self.variables_.remaining_times[
            task_id * self.instance.n_machines + machine_id
        ]

    def get_assignment(self, task_id: TaskID) -> MachineID:
        return self.variables_.assignment[task_id]

    def get_machines(self, task_id: TaskID) -> list[MachineID]:
        return [
            machine_id
            for machine_id in self.instance.processing_times[task_id]
            if self.is_feasible(task_id, machine_id)
        ]

    # TODO: Cache feasibility results to avoid redundant checks
    # 6.7% of the time is spent in this method
    def is_feasible(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:
        if not self.variables_.feasible[task_id]:
            return False

        elif machine_id == GLOBAL_MACHINE_ID:
            return True # We don't need to check 

        idx = task_id * self.instance.n_machines + machine_id
        lb = self.variables_.start.lbs[idx]
        ub = self.variables_.start.ubs[idx]

        return lb == ub or (lb < ub and self.time <= ub)

    ## Setter methods for variable values, triggering constraint propagation through events
    def require_task(self, task_id: TaskID) -> None:
        event = self.variables_.restrict_presence(task_id, PRESENT)
        if event is not None:
            self.event_queue.append(event)

    def forbid_task(self, task_id: TaskID) -> None:
        event = self.variables_.restrict_presence(task_id, ABSENT)
        if event is not None:
            self.event_queue.append(event)

    def tight_start_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        event = self.variables_.set_start_lb(task_id, value, machine_id)

        if event:
            self.event_queue.append(event)

    def tight_start_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        event = self.variables_.set_start_ub(task_id, value, machine_id)

        if event:
            self.event_queue.append(event)

    def tight_end_lb(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        event = self.variables_.set_end_lb(task_id, value, machine_id)

        if event:
            self.event_queue.append(event)

    def tight_end_ub(
        self,
        task_id: TaskID,
        value: Time,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        event = self.variables_.set_end_ub(task_id, value, machine_id)

        if event:
            self.event_queue.append(event)


    def forbid_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        event = self.variables_.restrict_machine(task_id, machine_id)

        if event:
            self.event_queue.append(event)

    def require_machine(self, task_id: TaskID, machine_id: MachineID) -> None:
        for other_machine in self.instance.processing_times[task_id]:
            if other_machine != machine_id:
                self.forbid_machine(task_id, other_machine)

    # Discrete event simulation API methods
    def is_awaiting(self, task_id: TaskID) -> bool:
        return task_id in self.runtime_state.awaiting_tasks

    def is_paused(self, task_id: TaskID) -> bool:
        return task_id in self.runtime_state.awaiting_tasks and bool(
            self.runtime_state.history[task_id]
        )

    def is_executing(self, task_id: TaskID) -> bool:
        return task_id in self.runtime_state.executing_tasks

    def is_completed(self, task_id: TaskID) -> bool:
        return task_id in self.runtime_state.completed_tasks

    def is_available(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> bool:
        vars_ = self.variables_

        start_var = vars_.start
        t = self.time

        lb = start_var.get_lb(task_id, machine_id)
        if lb > t:
            return False

        ub = start_var.get_ub(task_id, machine_id)

        return t < ub

    def execute_task(self, task_id: TaskID, machine_id: MachineID) -> None:
        start_time = self.time

        event = self.variables_.assign(task_id, start_time, machine_id)
        self.event_queue.append(event)

        end_time = start_time + self.get_remaining_time(task_id, machine_id)
        self.runtime_state.start_task(task_id, machine_id, start_time, end_time)

    def pause_task(self, task_id: TaskID) -> None:
        last_start = self.variables_.start.global_lbs[task_id]
        expected_end = self.variables_.end.global_ubs[task_id]

        if self.time >= expected_end:
            return

        expected_duration = expected_end - last_start
        actual_duration = self.time - last_start
        remaining_times = self.variables_.remaining_times

        start_idx = task_id * self.instance.n_machines
        for machine in self.instance.processing_times[task_id]:
            idx = start_idx + machine

            work_done = ((actual_duration) * remaining_times[idx]) // (
                expected_duration
            )
            remaining_times[idx] -= work_done

            self.variables_.start.lbs[idx] = self.time
            self.variables_.start.ubs[idx] = MAX_TIME

        self.variables_.start.global_lbs[task_id] = self.time
        self.variables_.start.global_ubs[task_id] = MAX_TIME

        self.variables_.assignment[task_id] = GLOBAL_MACHINE_ID
        self.runtime_state.pause_task(task_id, self.time)

    def get_next_available_time(self) -> Time:
        next_time = MAX_TIME

        for task_id in self.runtime_state.awaiting_tasks:
            start_lb = self.variables_.start.global_lbs[task_id]

            if start_lb < next_time:
                next_time = start_lb

        return next_time

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

        task_obs["status"] = self.runtime_state.status.copy()
        task_obs["available"] = [
            self.is_available(task_id) for task_id in range(self.n_tasks)
        ]

        return task_obs, {}

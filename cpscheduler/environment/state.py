from typing import Any
from typing_extensions import TypeAlias

from cpscheduler.environment.constants import (
    MAX_TIME,
    MIN_TIME,
    MACHINE_ID,
    TASK_ID,
    TIME,
    GLOBAL_MACHINE_ID,
    Presence
)
from cpscheduler.environment.events import Event, VarField
from cpscheduler.environment.tasks import (
    ProblemInstance,
    ScheduleVariables,
    RuntimeState
)

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

    time: TIME
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
    def n_tasks(self) -> int:
        return self.instance.n_tasks

    @property
    def n_jobs(self) -> int:
        return self.instance.n_jobs

    # TODO: Reduce the hit rate of this property (1.5% of the time is spent here)
    @property
    def n_machines(self) -> int:
        return self.instance.n_machines

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

    def advance_time(self, new_time: TIME) -> None:
        assert new_time >= self.time, "Cannot go back in time."

        self.time = new_time
        self.runtime_state.update(self.time)

    # Constraint propagation API methods

    ## Getter methods for variable values
    def get_start_lb(
        self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> TIME:
        return self.variables_.start.get_lb(task_id, machine_id)

    def get_start_ub(
        self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> TIME:
        return self.variables_.start.get_ub(task_id, machine_id)

    def get_end_lb(
        self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> TIME:
        return self.variables_.end.get_lb(task_id, machine_id)

    def get_end_ub(
        self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> TIME:
        return self.variables_.end.get_ub(task_id, machine_id)

    def get_remaining_time(
        self, task_id: TASK_ID, machine_id: MACHINE_ID
    ) -> TIME:
        return self.variables_.remaining_times[
            task_id * self.n_machines + machine_id
        ]

    def is_fixed(self, task_id: TASK_ID) -> bool:
        return self.variables_.presence[task_id] == Presence.FIXED

    def get_assignment(self, task_id: TASK_ID) -> MACHINE_ID:
        return self.variables_.assignment[task_id]

    def get_machines(self, task_id: TASK_ID) -> list[MACHINE_ID]:
        return [
            machine_id for machine_id in self.instance.processing_times[task_id]
            if self.is_feasible(task_id, machine_id)
        ]

    def is_present(self, task_id: TASK_ID) -> bool:
        return self.variables_.presence[task_id] == Presence.PRESENT

    # TODO: Cache feasibility results to avoid redundant checks
    # 6.7% of the time is spent in this method
    def is_feasible(
        self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> bool:   
        if machine_id != GLOBAL_MACHINE_ID:
            idx = task_id * self.n_machines + machine_id
            lb = self.variables_.start.lbs[idx]
            ub = self.variables_.start.ubs[idx]

            return lb == ub or (lb < ub and self.time <= ub)
        
        return self.variables_.presence[task_id] != Presence.INFEASIBLE


    def is_consistent(self, task_id: TASK_ID) -> bool:
        return not self.is_present(task_id) or self.is_feasible(task_id)

    ## Setter methods for variable values, triggering constraint propagation through events

    # This method is very inneficient when no infesibility is possible
    # TODO: Implement a more efficient method for checking feasibility after tightening bounds
    # def _check_state_feasibility(self, task_id: TASK_ID) -> None:
    #     if not self.is_feasible(task_id):
    #         if not self.instance.is_optional(task_id):
    #             self.infeasible = True
    #             return

    #         if self.is_present(task_id):
    #             self.variables_.presence[task_id] = Presence.ABSENT
    #             self.awaiting_tasks.discard(task_id)
    #             self.event_queue.append(
    #                 Event(task_id, VarField.ABSENCE, GLOBAL_MACHINE_ID)
    #             )

    def tight_start_lb(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        if value <= self.variables_.start.get_lb(task_id, machine_id):
            return

        self.variables_.set_start_lb(task_id, value, machine_id)
        self.event_queue.append(Event(task_id, VarField.START_LB, machine_id))
        # self._check_state_feasibility(task_id) # Expensive here

    def tight_start_ub(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        if value >= self.variables_.start.get_ub(task_id, machine_id):
            return

        self.variables_.set_start_ub(task_id, value, machine_id)
        self.event_queue.append(Event(task_id, VarField.START_UB, machine_id))
        # self._check_state_feasibility(task_id)

    def tight_end_lb(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        if value <= self.variables_.end.get_lb(task_id, machine_id):
            return

        self.variables_.set_end_lb(task_id, value, machine_id)
        self.event_queue.append(Event(task_id, VarField.END_LB, machine_id))
        # self._check_state_feasibility(task_id)

    def tight_end_ub(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:

        if value >= self.variables_.end.get_ub(task_id, machine_id):
            return

        self.variables_.set_end_ub(task_id, value, machine_id)
        self.event_queue.append(Event(task_id, VarField.END_UB, machine_id))
        # self._check_state_feasibility(task_id)

    def forbid_machine(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        self.tight_start_lb(task_id, MAX_TIME, machine_id)
        self.tight_start_ub(task_id, MIN_TIME, machine_id)

    # def require_task(self, task_id: TASK_ID) -> None:
    #     if self.variables_.presence[task_id] != Presence.PRESENT:
    #         self.variables_.presence[task_id] = Presence.PRESENT
    #         self.awaiting_tasks.add(task_id)
    #         self.event_queue.append(
    #             Event(task_id, VarField.PRESENCE, GLOBAL_MACHINE_ID)
    #         )

    def forbid_task(self, task_id: TASK_ID) -> None:
        self.forbid_machine(task_id, GLOBAL_MACHINE_ID)

    def require_machine(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        for other_machine in self.instance.processing_times[task_id]:
            if other_machine != machine_id:
                self.forbid_machine(task_id, other_machine)

    # Discrete event simulation API methods
    def is_awaiting(self, task_id: TASK_ID) -> bool:
        return task_id in self.runtime_state.awaiting_tasks

    def is_paused(self, task_id: TASK_ID) -> bool:
        return (
            task_id in self.runtime_state.awaiting_tasks and
            bool(self.runtime_state.history[task_id])
        )

    def is_executing(self, task_id: TASK_ID) -> bool:
        return task_id in self.runtime_state.executing_tasks

    def is_completed(self, task_id: TASK_ID) -> bool:
        return task_id in self.runtime_state.completed_tasks

    def is_available(
        self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> bool:
        vars_ = self.variables_

        if vars_.presence[task_id] != Presence.PRESENT:
            return False

        start_var = vars_.start
        t = self.time

        lb = start_var.get_lb(task_id, machine_id)
        if lb > t:
            return False

        ub = start_var.get_ub(task_id, machine_id)

        return t < ub

    def execute_task(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        start_vars = self.variables_.start
        end_vars = self.variables_.end

        remaining_time = self.variables_.remaining_times[
            task_id * self.n_machines + machine_id
        ]

        start_time = self.time
        end_time = start_time + remaining_time

        for machine in self.instance.processing_times[task_id]:
            idx = task_id * self.n_machines + machine

            if machine == machine_id:
                start_vars.lbs[idx] = start_time
                start_vars.ubs[idx] = start_time
                end_vars.lbs[idx] = end_time
                end_vars.ubs[idx] = end_time
            
            else:
                start_vars.lbs[idx] = MAX_TIME
                start_vars.ubs[idx] = MIN_TIME
                end_vars.lbs[idx] = MAX_TIME
                end_vars.ubs[idx] = MIN_TIME
        
        self.variables_.presence[task_id] = Presence.FIXED
        self.variables_.assignment[task_id] = machine_id
        start_vars.global_lbs[task_id] = start_time
        start_vars.global_ubs[task_id] = start_time

        self.event_queue.append(Event(task_id, VarField.ASSIGNMENT, machine_id))

        self.runtime_state.start_task(
            task_id,
            machine_id,
            start_time,
            end_time
        )

    def pause_task(self, task_id: TASK_ID) -> None:
        last_start = self.variables_.start.global_lbs[task_id]
        expected_end = self.variables_.end.global_ubs[task_id]

        if self.time >= expected_end:
            return

        expected_duration = expected_end - last_start
        actual_duration = self.time - last_start
        remaining_times = self.variables_.remaining_times

        for machine in self.instance.processing_times[task_id]:
            idx = task_id * self.n_machines + machine

            work_done = ((actual_duration) * remaining_times[idx]) // (
                expected_duration
            )
            remaining_times[idx] -= work_done

            # TODO: Produce an event instead of directly modifying the bounds
            idx = task_id * self.n_machines + machine

            self.variables_.start.lbs[idx] = self.time
            self.variables_.start.ubs[idx] = MAX_TIME

        self.variables_.start.global_lbs[task_id] = self.time
        self.variables_.start.global_ubs[task_id] = MAX_TIME

        self.variables_.assignment[task_id] = GLOBAL_MACHINE_ID
        self.runtime_state.pause_task(task_id, self.time)

    def get_next_available_time(self, strict: bool = False) -> TIME:
        next_time = MAX_TIME

        for task_id in self.runtime_state.awaiting_tasks:
            task_lb = self.variables_.start.global_lbs[task_id]

            if strict and task_lb <= self.time:
                continue

            if task_lb < next_time:
                next_time = task_lb

        return next_time

    def get_next_completion_time(self) -> TIME:
        next_time = MIN_TIME

        for task_id in self.runtime_state.executing_tasks:
            end_time = self.variables_.end.global_ubs[task_id]

            if end_time > next_time:
                next_time = end_time

        return next_time

    def get_machine_execution(self) -> dict[MACHINE_ID, list[TASK_ID]]:
        assignments: dict[MACHINE_ID, list[TASK_ID]] = {
            machine_id: [] for machine_id in range(self.n_machines)
        }

        for task_id in self.runtime_state.executing_tasks:
            machine_id = self.variables_.assignment[task_id]
            assignments[machine_id].append(task_id)

        return assignments

    def get_observation(self) -> ObsType:
        task_obs = self.instance.task_instance.copy()

        task_obs["status"] = self.runtime_state.status.copy()
        task_obs["available"] = [
            self.is_available(task_id) for task_id in range(self.n_tasks)
        ]

        return task_obs, {}

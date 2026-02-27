from typing import Any
from typing_extensions import TypeAlias

from cpscheduler.environment.constants import (
    MAX_TIME,
    MIN_TIME,
    MACHINE_ID,
    TASK_ID,
    TIME,
    GLOBAL_MACHINE_ID,
    Status,
    StatusType,
    Presence
)
from cpscheduler.environment.events import Event, VarField
from cpscheduler.environment.tasks import (
    ProblemInstance,
    ScheduleVariables,
    TaskHistory,
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
        "event_queue",
        "infeasible",
        "awaiting_tasks",
        "fixed_tasks",
        "task_history",
        "loaded",
    )

    instance: ProblemInstance

    time: TIME
    "Current simulation time."

    variables_: ScheduleVariables
    "Encapsulates all the variable data for the tasks"

    task_history: list[list[TaskHistory]]
    "History of task executions, indexed by task_id. Each entry is a list of TaskHistory."

    awaiting_tasks: set[TASK_ID]
    "Set of tasks that are awaiting execution. Does not include tasks that are set to be absent."

    fixed_tasks: set[TASK_ID]
    "Set of tasks that have been fixed in the schedule."

    event_queue: list[Event]
    "Queue of events to be processed for constraint propagation."

    loaded: bool
    infeasible: bool

    def __init__(self) -> None:
        self.time = 0

        self.awaiting_tasks = set()
        self.fixed_tasks = set()

        self.task_history = []

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

        self.variables_ = ScheduleVariables(self.instance)
        self.task_history = [[] for _ in range(self.n_tasks)]

        self.time = 0

        self.awaiting_tasks.update(range(self.n_tasks))

        self.event_queue.clear()
        self.fixed_tasks.clear()

    def is_terminal(self) -> bool:
        if self.awaiting_tasks:
            return self.infeasible

        return all(
            self.task_history[task_id][-1].end_time <= self.time
            for task_id in self.fixed_tasks
        )

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

    def get_assignment(self, task_id: TASK_ID) -> MACHINE_ID:
        return self.variables_.assignment[task_id]

    def is_present(self, task_id: TASK_ID) -> bool:
        return self.variables_.presence[task_id] == Presence.PRESENT

    # TODO: Cache feasibility results to avoid redundant checks
    # 6.7% of the time is spent in this method
    def is_feasible(
        self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> bool:   
        start = self.variables_.start

        lb = start.get_lb(task_id, machine_id)
        ub = start.get_ub(task_id, machine_id)

        if lb > ub or self.time > ub:
            return False

        for machine in self.instance.processing_times[task_id]:
            idx = task_id * self.n_machines + machine

            if self.variables_.start.lbs[idx] > self.variables_.start.ubs[idx]:
                return False

        return True

    def is_consistent(self, task_id: TASK_ID) -> bool:
        return not self.is_present(task_id) or self.is_feasible(task_id)

    ## Setter methods for variable values, triggering constraint propagation through events

    # This method is very inneficient when no infesibility is possible
    # TODO: Implement a more efficient method for checking feasibility after tightening bounds
    def _check_state_feasibility(self, task_id: TASK_ID) -> None:
        if not self.is_feasible(task_id):
            if not self.instance.is_optional(task_id):
                self.infeasible = True
                return

            if self.is_present(task_id):
                self.variables_.presence[task_id] = Presence.ABSENT
                self.awaiting_tasks.discard(task_id)
                self.event_queue.append(
                    Event(task_id, VarField.ABSENCE, GLOBAL_MACHINE_ID)
                )

    def tight_start_lb(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        start_vars = self.variables_.start

        if value <= start_vars.get_lb(task_id, machine_id):
            return

        end_vars = self.variables_.end

        if machine_id == GLOBAL_MACHINE_ID:
            remaining_times = self.variables_.remaining_times

            for machine in self.instance.processing_times[task_id]:
                idx = task_id * self.n_machines + machine

                if start_vars.lbs[idx] < value:
                    start_vars.lbs[idx] = value
                    end_vars.lbs[idx] = value + remaining_times[idx]

            start_vars.global_lbs[task_id] = value
            end_vars.recompute_global_bounds(task_id) # Expensive here

        else:
            idx = task_id * self.n_machines + machine_id

            start_vars.lbs[idx] = value
            end_vars.lbs[idx] = value + self.variables_.remaining_times[idx]

            start_vars.recompute_global_bounds(task_id)
            end_vars.recompute_global_bounds(task_id)

        self.event_queue.append(Event(task_id, VarField.START_LB, machine_id))
        self._check_state_feasibility(task_id) # Expensive here

    def tight_start_ub(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        start_vars = self.variables_.start

        if value >= start_vars.get_ub(task_id, machine_id):
            return

        end_vars = self.variables_.end

        if machine_id == GLOBAL_MACHINE_ID:
            remaining_times = self.variables_.remaining_times

            for machine in self.instance.processing_times[task_id]:
                idx = task_id * self.n_machines + machine

                if start_vars.ubs[idx] > value:
                    start_vars.ubs[idx] = value
                    end_vars.ubs[idx] = value + remaining_times[idx]

            start_vars.global_ubs[task_id] = value
            end_vars.recompute_global_bounds(task_id)

        else:
            idx = task_id * self.n_machines + machine_id

            start_vars.ubs[idx] = value
            end_vars.ubs[idx] = value + self.variables_.remaining_times[idx]

            start_vars.recompute_global_bounds(task_id)
            end_vars.recompute_global_bounds(task_id)

        self.event_queue.append(Event(task_id, VarField.START_UB, machine_id))
        self._check_state_feasibility(task_id)

    def tight_end_lb(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        end_vars = self.variables_.end

        if value <= end_vars.get_lb(task_id, machine_id):
            return

        start_vars = self.variables_.start

        if machine_id == GLOBAL_MACHINE_ID:
            remaining_times = self.variables_.remaining_times

            for machine in self.instance.processing_times[task_id]:
                idx = task_id * self.n_machines + machine

                if end_vars.lbs[idx] < value:
                    end_vars.lbs[idx] = value
                    start_vars.lbs[idx] = value - remaining_times[idx]

            end_vars.global_lbs[task_id] = value
            start_vars.recompute_global_bounds(task_id)

        else:
            idx = task_id * self.n_machines + machine_id

            end_vars.lbs[idx] = value
            start_vars.lbs[idx] = value - self.variables_.remaining_times[idx]

            end_vars.recompute_global_bounds(task_id)
            start_vars.recompute_global_bounds(task_id)

        self.event_queue.append(Event(task_id, VarField.END_LB, machine_id))
        self._check_state_feasibility(task_id)

    def tight_end_ub(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        end_vars = self.variables_.end

        if value >= end_vars.get_ub(task_id, machine_id):
            return

        start_vars = self.variables_.start

        if machine_id == GLOBAL_MACHINE_ID:
            remaining_times = self.variables_.remaining_times

            for machine in self.instance.processing_times[task_id]:
                idx = task_id * self.n_machines + machine

                if end_vars.ubs[idx] > value:
                    end_vars.ubs[idx] = value
                    start_vars.ubs[idx] = value - remaining_times[idx]

            end_vars.global_ubs[task_id] = value
            start_vars.recompute_global_bounds(task_id)

        else:
            idx = task_id * self.n_machines + machine_id

            end_vars.ubs[idx] = value
            start_vars.ubs[idx] = value - self.variables_.remaining_times[idx]

            end_vars.recompute_global_bounds(task_id)
            start_vars.recompute_global_bounds(task_id)

        self.event_queue.append(Event(task_id, VarField.END_UB, machine_id))
        self._check_state_feasibility(task_id)

    def forbid_machine(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        self.tight_start_lb(task_id, MAX_TIME, machine_id)
        self.tight_start_ub(task_id, MIN_TIME, machine_id)

    def require_task(self, task_id: TASK_ID) -> None:
        if self.variables_.presence[task_id] != Presence.PRESENT:
            self.variables_.presence[task_id] = Presence.PRESENT
            self.awaiting_tasks.add(task_id)
            self.event_queue.append(
                Event(task_id, VarField.PRESENCE, GLOBAL_MACHINE_ID)
            )

    def forbid_task(self, task_id: TASK_ID) -> None:
        self.forbid_machine(task_id, GLOBAL_MACHINE_ID)

    def require_machine(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        for other_machine in self.instance.processing_times[task_id]:
            if other_machine != machine_id:
                self.forbid_machine(task_id, other_machine)

    # Discrete event simulation API methods
    def is_fixed(self, task_id: TASK_ID) -> bool:
        return task_id in self.fixed_tasks

    def is_awaiting(self, task_id: TASK_ID) -> bool:
        return task_id in self.awaiting_tasks

    def is_paused(self, task_id: TASK_ID) -> bool:
        history = self.task_history[task_id]

        if not history or task_id in self.fixed_tasks:
            return False

        return history[-1].end_time <= self.time

    def is_executing(self, task_id: TASK_ID) -> bool:
        history = self.task_history[task_id]

        if not history:
            return False

        return self.time < history[-1].end_time

    def is_completed(self, task_id: TASK_ID) -> bool:
        history = self.task_history[task_id]

        if not history or task_id in self.awaiting_tasks:
            return False

        return self.task_history[task_id][-1].end_time <= self.time

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

    def get_status(self, task_id: TASK_ID) -> StatusType:
        "Get the current status of the task at a given time."
        lb = self.variables_.start.global_lbs[task_id]
        ub = self.variables_.start.global_ubs[task_id]

        if ub < lb:
            return Status.INFEASIBLE

        if self.time < ub:
            return Status.AWAITING

        if self.is_executing(task_id):
            return Status.EXECUTING

        if self.is_paused(task_id):
            return Status.PAUSED

        return Status.COMPLETED

    def execute_task(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        self.tight_start_lb(task_id, self.time, machine_id)
        self.tight_start_ub(task_id, self.time, machine_id)

        self.variables_.assignment[task_id] = machine_id

        self.awaiting_tasks.remove(task_id)
        self.fixed_tasks.add(task_id)

        start = self.variables_.start.get_lb(task_id, machine_id)
        end = self.variables_.end.get_ub(task_id, machine_id)

        history_entry = TaskHistory(
            assignment=machine_id, start_time=start, duration=end - start
        )

        self.task_history[task_id].append(history_entry)

    def pause_task(self, task_id: TASK_ID) -> None:
        history_entry = self.task_history[task_id][-1]

        expected_duration = history_entry.duration
        actual_duration = self.time - history_entry.start_time

        if actual_duration >= expected_duration:
            return

        self.task_history[task_id].pop()

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

        self.task_history[task_id].append(
            TaskHistory(
                assignment=history_entry.assignment,
                start_time=history_entry.start_time,
                duration=actual_duration,
            )
        )

        self.awaiting_tasks.add(task_id)
        self.fixed_tasks.remove(task_id)

    def get_next_available_time(self, strict: bool = False) -> TIME:
        next_time = MAX_TIME

        for task_id in self.awaiting_tasks:
            task_lb = self.variables_.start.global_lbs[task_id]

            if strict and task_lb <= self.time:
                continue

            if task_lb < next_time:
                next_time = task_lb

        return next_time

    def get_next_completion_time(self) -> TIME:
        next_time = MIN_TIME

        for task_id in self.fixed_tasks:
            end_time = self.variables_.end.global_ubs[task_id]

            if end_time > next_time:
                next_time = end_time

        return next_time

    def get_machine_execution(self) -> dict[MACHINE_ID, list[TASK_ID]]:
        assignments: dict[MACHINE_ID, list[TASK_ID]] = {
            machine_id: [] for machine_id in range(self.n_machines)
        }

        for task_id in self.fixed_tasks:
            if self.is_executing(task_id):
                machine_id = self.variables_.assignment[task_id]
                assignments[machine_id].append(task_id)

        return assignments

    def get_observation(self) -> ObsType:
        task_obs = self.instance.task_instance.copy()

        task_obs["status"] = [
            self.get_status(task_id) for task_id in range(self.n_tasks)
        ]
        task_obs["available"] = [
            self.is_available(task_id) for task_id in range(self.n_tasks)
        ]

        return task_obs, {}

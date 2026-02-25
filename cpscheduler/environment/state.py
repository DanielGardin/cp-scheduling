from typing import Any

from cpscheduler.environment._common import (
    MAX_TIME,
    MIN_TIME,
    MACHINE_ID,
    TASK_ID,
    TIME,
    ObsType,
    GLOBAL_MACHINE_ID,
    Status,
    StatusType
)
from cpscheduler.environment.events import Event, VarField
from cpscheduler.environment.tasks import Task, Job, TaskHistory, ScheduleVariables
from cpscheduler.utils.list_utils import convert_to_list


def check_instance_consistency(instance: dict[str, list[Any]]) -> int:
    "Check if all lists in the instance have the same length."
    lengths = {len(v) for v in instance.values()}

    if len(lengths) > 1:
        raise ValueError("Inconsistent instance data: all lists must have the same length.")

    return lengths.pop() if lengths else 0


class ScheduleState:
    """
    ScheduleState represents the current state of the scheduling environment, working as both a
    Discrete Event Simulation (DES) state and a Constraint Satisfaction Problem (CSP) state.

    It has no simulation logic itself, instead, it provides an API to read and modify the current
    state of the problem. The actual kernel is implemented in the SchedulingEnv class,
    which knows the constraints and how to propagate changes through them.
    """
    __slots__ = (
        "tasks",
        "jobs",
        "time",
        "variables_",
        "task_history",
        "awaiting_tasks",
        "fixed_tasks",
        "event_queue",
        "instance",
        "job_instance",
        "_n_machines",
        "infeasible"
    )

    tasks: list[Task]
    "Static list of all tasks in the scheduling problem."

    jobs: list[Job]
    "Static list of all jobs in the scheduling problem."

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

    instance: dict[str, list[Any]]
    job_instance: dict[str, list[Any]]
    _n_machines: int

    infeasible: bool

    def __init__(self) -> None:
        self.tasks = []
        self.jobs = []

        self.time = 0

        self.awaiting_tasks = set()
        self.fixed_tasks = set()

        self.task_history = []

        self.event_queue = []

        self.instance = {}
        self.job_instance = {}
        self._n_machines = 0

        self.infeasible = False

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            (),
            (
                self.tasks,
                self.jobs,
                self.time,
                self.variables_,
                self.task_history,
                self.awaiting_tasks,
                self.fixed_tasks,
                self.event_queue,
                self.instance,
                self.job_instance,
                self._n_machines,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.tasks,
            self.jobs,
            self.time,
            self.variables_,
            self.task_history,
            self.awaiting_tasks,
            self.fixed_tasks,
            self.event_queue,
            self.instance,
            self.job_instance,
            self._n_machines,
        ) = state

    # Properties
    @property
    def n_machines(
        self,
    ) -> int:  # Lazy evaluation of number of machines, only works after initialization
        if self._n_machines > 0 or not self.loaded:
            return self._n_machines

        max_machine_id = -1
        for task in self.tasks:
            for machine in task.machines:
                if machine > max_machine_id:
                    max_machine_id = machine

        self._n_machines = max_machine_id + 1
        return self._n_machines

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    @property
    def loaded(self) -> bool:
        return bool(self.tasks)

    # Instance control methods
    def clear(self) -> None:
        self.tasks.clear()
        self.jobs.clear()

        self.time = 0
        self.event_queue.clear()

        self.task_history.clear()

        self.awaiting_tasks.clear()
        self.fixed_tasks.clear()

        self.instance.clear()
        self.job_instance.clear()
        self._n_machines = 0

    def read_instance(
        self,
        task_data: dict[str, list[Any]],
    ) -> None:
        self.clear()

        self.instance = task_data
        self.job_instance = {}
        n_tasks = check_instance_consistency(task_data)

        job_ids: list[TASK_ID]
        if "job" in task_data:
            job_ids = convert_to_list(task_data["job"], TASK_ID)

        else:
            job_ids = list(range(n_tasks))

        n_jobs = max(job_ids) + 1
        for job_id in range(n_jobs):
            self.jobs.append(Job(job_id))

        for task_id, job_id in enumerate(job_ids):
            task = Task(task_id, job_id)

            self.tasks.append(task)
            self.jobs[job_id].add_task(task)

        self.instance["task_id"] = list(range(n_tasks))
        self.instance["job_id"] = job_ids
        self.job_instance["job_id"] = list(range(self.n_jobs))

    # Flow control methods
    def reset(self) -> None:
        self.variables_ = ScheduleVariables(self.tasks, self.n_machines)
        self.task_history = [[] for _ in range(self.n_tasks)]

        self.time = 0

        self.awaiting_tasks.update(range(self.n_tasks))

        self.event_queue.clear()
        self.fixed_tasks.clear()

    def is_terminal(self) -> bool:
        if self.awaiting_tasks:
            return self.infeasible

        return all(
            self.task_history[task_id][-1].end_time <= self.time for task_id in self.fixed_tasks
        )

    # Constraint propagation API methods

    ## Getter methods for variable values
    def get_start_lb(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        return self.variables_.start.get_lb(task_id, machine_id)

    def get_start_ub(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        return self.variables_.start.get_ub(task_id, machine_id)

    def get_end_lb(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        return self.variables_.end.get_lb(task_id, machine_id)

    def get_end_ub(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        return self.variables_.end.get_ub(task_id, machine_id)

    def get_remaining_time(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> TIME:
        return self.variables_.remaining_times[task_id * self.n_machines + machine_id]

    def get_assignment(self, task_id: TASK_ID) -> MACHINE_ID:
        return self.variables_.assignment[task_id]

    def is_present(self, task_id: TASK_ID) -> bool:
        return self.variables_.present[task_id]

    def is_feasible(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> bool:
        start = self.variables_.start

        lb = start.get_lb(task_id, machine_id)
        ub = start.get_ub(task_id, machine_id)

        if lb > ub or self.time > ub:
            return False

        for machine in self.tasks[task_id].machines:
            idx = task_id * self.n_machines + machine

            if self.variables_.start.lbs[idx] > self.variables_.start.ubs[idx]:
                return False

        return True

    def is_consistent(self, task_id: TASK_ID) -> bool:
        return not self.variables_.present[task_id] or self.is_feasible(task_id)

    ## Setter methods for variable values, triggering constraint propagation through events
    def _check_state_feasibility(self, task_id: TASK_ID) -> None:
        if not self.is_feasible(task_id):
            if not self.tasks[task_id].optional:
                self.infeasible = True
                return

            if self.variables_.present[task_id]:
                self.variables_.present[task_id] = False
                self.awaiting_tasks.discard(task_id)
                self.event_queue.append(Event(task_id, VarField.ABSENCE, GLOBAL_MACHINE_ID))

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

            for machine in self.tasks[task_id].machines:
                idx = task_id * self.n_machines + machine

                if start_vars.lbs[idx] < value:
                    start_vars.lbs[idx] = value
                    end_vars.lbs[idx] = value + remaining_times[idx]

            start_vars.global_lbs[task_id] = value
            end_vars.recompute_global_bounds(task_id)

        else:
            idx = task_id * self.n_machines + machine_id

            start_vars.lbs[idx] = value
            end_vars.lbs[idx] = value + self.variables_.remaining_times[idx]

            start_vars.recompute_global_bounds(task_id)
            end_vars.recompute_global_bounds(task_id)


        self.event_queue.append(Event(task_id, VarField.START_LB, machine_id))
        self._check_state_feasibility(task_id)

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

            for machine in self.tasks[task_id].machines:
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

            for machine in self.tasks[task_id].machines:
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

            for machine in self.tasks[task_id].machines:
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
        if not self.variables_.present[task_id]:
            self.variables_.present[task_id] = True
            self.awaiting_tasks.add(task_id)
            self.event_queue.append(Event(task_id, VarField.PRESENCE, GLOBAL_MACHINE_ID))

    def forbid_task(self, task_id: TASK_ID) -> None:
        self.forbid_machine(task_id, GLOBAL_MACHINE_ID)

    def require_machine(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        for other_machine in self.tasks[task_id].machines:
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

    def is_available(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> bool:
        vars_ = self.variables_

        if not vars_.present[task_id]:
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

        history_entry = TaskHistory(assignment=machine_id, start_time=start, duration=end - start)

        self.task_history[task_id].append(history_entry)

    def pause_task(self, task_id: TASK_ID) -> None:
        history_entry = self.task_history[task_id][-1]

        expected_duration = history_entry.duration
        actual_duration = self.time - history_entry.start_time

        if actual_duration >= expected_duration:
            return

        self.task_history[task_id].pop()

        remaining_times = self.variables_.remaining_times

        for machine in self.tasks[task_id].machines:
            idx = task_id * self.n_machines + machine

            work_done = ((actual_duration) * remaining_times[idx]) // (expected_duration)
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
        task_obs = self.instance.copy()
        job_obs = self.job_instance.copy()

        task_obs["status"] = [self.get_status(task_id) for task_id in range(self.n_tasks)]
        task_obs["available"] = [self.is_available(task_id) for task_id in range(self.n_tasks)]

        return task_obs, job_obs

from typing import Any
from collections import deque

from cpscheduler.environment._common import (
    MAX_TIME,
    MIN_TIME,
    MACHINE_ID,
    TASK_ID,
    TIME,
    ObsType,
    GLOBAL_MACHINE_ID,
    STATUS,
    StatusEnum,
)
from cpscheduler.environment.events import VarField, Event
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
    state of the problem. The actual simulation logic is implemented in the SchedulingEnv class,
    which knows the constraints and how to propagate changes through them.
    """

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
    "Set of tasks that are awaiting execution."

    fixed_tasks: set[TASK_ID]
    "Set of tasks that have been fixed in the schedule."

    event_queue: deque[Event]
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

        self.event_queue = deque()

        self.instance = {}
        self.job_instance = {}
        self._n_machines = 0

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
        return self.n_tasks > 0

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
        self.variables_ = ScheduleVariables(self.tasks)
        self.task_history = [[] for _ in range(self.n_tasks)]

        self.time = 0

        self.awaiting_tasks.update(range(self.n_tasks))

        self.event_queue.clear()
        self.fixed_tasks.clear()

    def is_terminal(self) -> bool:
        if self.awaiting_tasks:
            return any(
                self.is_present(task_id) and not self.is_feasible(task_id)
                for task_id in self.awaiting_tasks
            )

        return all(
            self.task_history[task_id][-1].end_time <= self.time for task_id in range(self.n_tasks)
        )

    # Constraint propagation API methods
    def get_bound(
        self, task_id: TASK_ID, field: VarField, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID
    ) -> TIME:
        return self.variables_.select_bound(task_id, machine_id, field)

    def get_start_lb(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.variables_.start.global_lbs[task_id]

        return self.variables_.start.lbs[task_id][machine_id]

    def get_start_ub(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.variables_.start.global_ubs[task_id]

        return self.variables_.start.ubs[task_id][machine_id]

    def get_end_lb(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.variables_.end.global_lbs[task_id]

        return self.variables_.end.lbs[task_id][machine_id]

    def get_end_ub(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.variables_.end.global_ubs[task_id]

        return self.variables_.end.ubs[task_id][machine_id]

    def get_remaining_time(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> TIME:
        return self.variables_.remaining_times[task_id][machine_id]

    def get_assignment(self, task_id: TASK_ID) -> MACHINE_ID:
        return self.variables_.assignment[task_id]

    def is_fixed(self, task_id: TASK_ID) -> bool:
        return self.variables_.fixed[task_id]

    def is_locked(self, task_id: TASK_ID) -> bool:
        return self.variables_.locked[task_id]

    def is_present(self, task_id: TASK_ID) -> bool:
        return self.variables_.present[task_id]

    def is_feasible(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> bool:
        return self.get_start_lb(task_id, machine_id) <= self.get_start_ub(
            task_id, machine_id
        ) and (self.is_fixed(task_id) or self.time <= self.get_start_ub(task_id, machine_id))

    def tight_start_lb(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        if self.is_fixed(task_id) or value <= self.get_start_lb(task_id, machine_id):
            return

        if machine_id == GLOBAL_MACHINE_ID:
            for machine in self.tasks[task_id].machines:
                if self.variables_.start.lbs[task_id][machine] < value:
                    self.variables_.start.lbs[task_id][machine] = value
                    self.variables_.end.lbs[task_id][machine] = (
                        value + self.variables_.remaining_times[task_id][machine]
                    )

            self.variables_.start.global_lbs[task_id] = value
            self.variables_.end.recompute_global_bounds(task_id)

        else:
            self.variables_.start.lbs[task_id][machine_id] = value
            self.variables_.end.lbs[task_id][machine_id] = (
                value + self.variables_.remaining_times[task_id][machine_id]
            )

            self.variables_.start.recompute_global_bounds(task_id)
            self.variables_.end.recompute_global_bounds(task_id)

        self.event_queue.append(Event(task_id, VarField.START_LB, machine_id))

    def tight_start_ub(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        if self.is_fixed(task_id) or value >= self.get_start_ub(task_id, machine_id):
            return

        if machine_id == GLOBAL_MACHINE_ID:
            for machine in self.tasks[task_id].machines:
                if self.variables_.start.ubs[task_id][machine] > value:
                    self.variables_.start.ubs[task_id][machine] = value
                    self.variables_.end.ubs[task_id][machine] = (
                        value + self.variables_.remaining_times[task_id][machine]
                    )

            self.variables_.start.global_ubs[task_id] = value
            self.variables_.end.recompute_global_bounds(task_id)

        else:
            self.variables_.start.ubs[task_id][machine_id] = value
            self.variables_.end.ubs[task_id][machine_id] = (
                value + self.variables_.remaining_times[task_id][machine_id]
            )

            self.variables_.start.recompute_global_bounds(task_id)
            self.variables_.end.recompute_global_bounds(task_id)

        self.event_queue.append(Event(task_id, VarField.START_UB, machine_id))

    def tight_end_lb(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        if self.is_fixed(task_id) or value <= self.get_end_lb(task_id, machine_id):
            return

        if machine_id == GLOBAL_MACHINE_ID:
            for machine in self.tasks[task_id].machines:
                if self.variables_.end.lbs[task_id][machine] < value:
                    self.variables_.end.lbs[task_id][machine] = value
                    self.variables_.start.lbs[task_id][machine] = (
                        value - self.variables_.remaining_times[task_id][machine]
                    )

            self.variables_.end.global_lbs[task_id] = value
            self.variables_.start.recompute_global_bounds(task_id)

        else:
            self.variables_.end.lbs[task_id][machine_id] = value
            self.variables_.start.lbs[task_id][machine_id] = (
                value - self.variables_.remaining_times[task_id][machine_id]
            )

            self.variables_.end.recompute_global_bounds(task_id)
            self.variables_.start.recompute_global_bounds(task_id)

        self.event_queue.append(Event(task_id, VarField.END_LB, machine_id))

    def tight_end_ub(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        if self.is_fixed(task_id) or value >= self.get_end_ub(task_id, machine_id):
            return

        if machine_id == GLOBAL_MACHINE_ID:
            for machine in self.tasks[task_id].machines:
                if self.variables_.end.ubs[task_id][machine] > value:
                    self.variables_.end.ubs[task_id][machine] = value
                    self.variables_.start.ubs[task_id][machine] = (
                        value - self.variables_.remaining_times[task_id][machine]
                    )

            self.variables_.end.global_ubs[task_id] = value
            self.variables_.start.recompute_global_bounds(task_id)

        else:
            self.variables_.end.ubs[task_id][machine_id] = value
            self.variables_.start.ubs[task_id][machine_id] = (
                value - self.variables_.remaining_times[task_id][machine_id]
            )

            self.variables_.end.recompute_global_bounds(task_id)
            self.variables_.start.recompute_global_bounds(task_id)

        self.event_queue.append(Event(task_id, VarField.END_UB, machine_id))

    def set_infeasible(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> None:
        self.tight_start_lb(task_id, MAX_TIME, machine_id)
        self.tight_start_ub(task_id, MIN_TIME, machine_id)

    def require_machine(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        for other_machine in self.tasks[task_id].machines:
            if other_machine != machine_id:
                self.set_infeasible(task_id, other_machine)

    def require_task(self, task_id: TASK_ID) -> None:
        if self.is_fixed(task_id) or self.is_present(task_id):
            return

        self.variables_.present[task_id] = True
        self.event_queue.append(Event(task_id, VarField.PRESENCE, GLOBAL_MACHINE_ID))

    def forbid_task(self, task_id: TASK_ID) -> None:
        if self.is_fixed(task_id) or not self.is_present(task_id):
            return

        self.set_infeasible(task_id)

        self.variables_.present[task_id] = False
        self.event_queue.append(Event(task_id, VarField.PRESENCE, GLOBAL_MACHINE_ID))

    # Discrete event simulation API methods
    # TODO: Remove is_available check and let the environment handle it, since it should be part of the constraints
    def is_awaiting(self, task_id: TASK_ID) -> bool:
        return not self.is_fixed(task_id)

    def is_paused(self, task_id: TASK_ID) -> bool:
        if not self.is_fixed(task_id):
            return bool(self.task_history[task_id])

        for i in range(len(self.task_history[task_id]) - 1, -1, -1):
            history_entry = self.task_history[task_id][i]

            if history_entry.end_time <= self.time:
                return True

            if history_entry.start_time <= self.time:
                return False

        return False

    def is_executing(self, task_id: TASK_ID) -> bool:
        if not self.is_fixed(task_id):
            return False

        for i in range(len(self.task_history[task_id]) - 1, -1, -1):
            history_entry = self.task_history[task_id][i]

            if history_entry.start_time <= self.time < history_entry.end_time:
                return True

        return False

    def is_completed(self, task_id: TASK_ID) -> bool:
        if not self.is_fixed(task_id) or not self.task_history[task_id]:
            return False

        return self.task_history[task_id][-1].end_time <= self.time

    def is_available(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID) -> bool:
        if self.is_fixed(task_id) or not self.is_present(task_id) or self.is_locked(task_id):
            return False

        start = self.get_start_lb(task_id, machine_id)
        end = self.get_start_ub(task_id, machine_id)

        return start <= self.time <= end

    def get_status(self, task_id: TASK_ID) -> STATUS:
        "Get the current status of the task at a given time."
        history = self.task_history[task_id]

        if self.is_completed(task_id):
            return StatusEnum.COMPLETED

        for idx in range(len(history) - 1, -1, -1):
            history_entry = history[idx]

            if history_entry.end_time <= self.time:
                return StatusEnum.PAUSED

            if history_entry.start_time <= self.time:
                return StatusEnum.EXECUTING

        if self.is_feasible(task_id):
            return StatusEnum.AWAITING

        return StatusEnum.INFEASIBLE

    def lock_task(self, task_id: TASK_ID) -> None:
        "Lock task from being executed, regardless of its bounds"
        self.variables_.locked[task_id] = True

    def unlock_task(self, task_id: TASK_ID) -> None:
        "Unlock task, allowing it to be executed if its bounds allow it"
        self.variables_.locked[task_id] = False

    def execute_task(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        self.tight_start_lb(task_id, self.time, machine_id)
        self.tight_start_ub(task_id, self.time, machine_id)

        self.variables_.assignment[task_id] = machine_id
        self.variables_.fixed[task_id] = True

        self.awaiting_tasks.remove(task_id)
        self.fixed_tasks.add(task_id)

        start = self.get_start_lb(task_id, machine_id)
        end = self.get_end_ub(task_id, machine_id)

        history_entry = TaskHistory(assignment=machine_id, start_time=start, duration=end - start)

        self.task_history[task_id].append(history_entry)

    def pause_task(self, task_id: TASK_ID) -> None:
        history_entry = self.task_history[task_id][-1]

        expected_duration = history_entry.duration
        actual_duration = self.time - history_entry.start_time

        if actual_duration >= expected_duration:
            return

        self.task_history[task_id].pop()

        remaining_times = self.variables_.remaining_times[task_id]

        for machine in self.tasks[task_id].machines:
            work_done = ((actual_duration) * remaining_times[machine]) // (expected_duration)

            self.variables_.remaining_times[task_id][machine] -= work_done

            # TODO: Produce an event instead of directly modifying the bounds
            self.variables_.start.lbs[task_id][machine] = self.time
            self.variables_.start.ubs[task_id][machine] = MAX_TIME

        self.variables_.start.global_lbs[task_id] = self.time
        self.variables_.start.global_ubs[task_id] = MAX_TIME

        self.variables_.fixed[task_id] = False
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
            task_lb = self.get_start_lb(task_id)

            if strict and task_lb <= self.time:
                continue

            if task_lb < next_time:
                next_time = task_lb

        return next_time

    def get_next_completion_time(self) -> TIME:
        next_time = MIN_TIME

        for task_id in self.fixed_tasks:
            end_time = self.get_end_lb(task_id)

            if end_time > next_time:
                next_time = end_time

        return next_time

    def get_machine_execution(self, time: TIME | None = None) -> dict[MACHINE_ID, list[TASK_ID]]:
        if time is None:
            time = self.time

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

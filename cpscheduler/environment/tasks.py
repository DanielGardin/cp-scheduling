"""
    tasks.py

This module contains the structs for manipulating the tasks in the scheduling environment.
In this current implementation, tasks are minimal units of work that can be scheduled and
it's defined by:
    - task_id: unique identifier for the task, usually its position in the list of tasks
    - processing_time: time required to complete the task
    - start_lbs: lower bounds for the starting time of the task
    - start_ubs: upper bounds for the starting time of the task
    - durations: time required to complete the task
    - assignments: machine assigned to the task

Tasks also accept preemption by default thought the pause method. This allows the task to
be split into multiple parts, each with its own starting time and duration.

The Tasks class is a container for the tasks and it's responsible for managing the tasks
and their states. This class is passed around the environment and heavily interacts with
every other classes in this library. It also stores additional data about the tasks that
can be used in constraints and objectives (i.e. due-dates, customer, type of task).

We do not reccomend customizing this module, as it's tightly coupled with the other modules
in the environment, change with caution.
"""
from typing import Any, NamedTuple, ClassVar
from collections.abc import Iterator

from mypy_extensions import u8

from ._common import (
    MIN_INT,
    MAX_INT,
    MACHINE_ID,
    TASK_ID,
    PART_ID,
    TIME,
)

JOB_ID_ALIASES = ["job", "job_id"]

class Status:
    "Possible statuses of a task at a given time."
    AWAITING : ClassVar[u8] = 0  # time < start_lb[0] or waiting for a machine
    EXECUTING: ClassVar[u8] = 1  # start_lb[i] <= time < start_lb[i] + duration[i] for some i
    PAUSED   : ClassVar[u8] = 2  # start_lb[i] + duration[i] < = time < start_lb[i+1] for some i
    COMPLETED: ClassVar[u8] = 3  # time >= start_lb[-1] + duration[-1]
    UNKNOWN  : ClassVar[u8] = 4  # unknown status


status_str = {
    Status.AWAITING : "awaiting",
    Status.EXECUTING: "executing",
    Status.PAUSED   : "paused",
    Status.COMPLETED: "completed",
    Status.UNKNOWN  : "unknown",
}

def ceil_div(a: TIME, b: TIME) -> TIME:
    "a divided by b, rounded up to the nearest integer."
    return -(-a // b)

class Bounds(NamedTuple):
    "Store the lower and upper bounds for decision variables."
    lb: TIME = 0
    ub: TIME = MAX_INT

class Task:
    """
    Minimal unit of work that can be scheduled. The task does not know anything about the
    environment, it only knows about its own state and the processing times for each machine.

    The task is defined by:
        - task_id: unique identifier for the task, usually its position in the list of tasks
        - processing_times: time required to complete the task
        - job: job id for the task
        - start_bounds: lower and upper bounds for the starting time of the task
        - durations: time required to complete the task
        - assignments: machine assigned to the task

    The environment has complete information about every task and orchestrates the scheduling
    process by modifying the task state.
    The task can be split into multiple parts, each with its own starting time and duration.
    """
    processing_times: dict[MACHINE_ID, TIME]

    starts     : list[TIME]
    durations  : list[TIME]
    assignments: list[MACHINE_ID]

    start_bounds: dict[MACHINE_ID, Bounds]

    n_parts: PART_ID

    def __init__(
        self,
        task_id: TASK_ID,
        processing_times: dict[MACHINE_ID, TIME]
    ) -> None:
        self.task_id = task_id
        self.processing_times = processing_times

        self.starts      = []
        self.durations   = []
        self.assignments = []

        self.start_bounds = {
            machine: Bounds() for machine in processing_times
        }

        self._remaining_times = self.processing_times.copy()

        self.fixed   = False
        self.n_parts = 0

    def __repr__(self) -> str:
        representation =  f"Task(id={self.task_id}"

        allocation_times = [
            f"[{self.starts[i]}, {self.starts[i] + self.durations[i]}] @ {self.assignments[i]}"
            for i in range(self.n_parts)
        ]

        if allocation_times:
            representation += f", {', '.join(allocation_times)}"

        else:
            representation += ", []"

        representation += ")"

        return representation

    def reset(self) -> None:
        "Resets the task to its initial state."
        self.fixed   = False
        self.n_parts = 0

        self.starts.clear()
        self.durations.clear()
        self.assignments.clear()

        self._new_part(0)

    def _new_part(self, time: TIME) -> None:
        for machine in self.processing_times:
            self.start_bounds[machine] = Bounds(
                lb=time,
            )

    @property
    def machines(self) -> list[MACHINE_ID]:
        "Get the list of machines that can process this task."
        return [machine for machine in self.processing_times]

    def is_fixed(self) -> bool:
        "Checks if the task has its decision variables fixed."
        return self.fixed

    def get_start(self, part: PART_ID = 0) -> TIME:
        "Get the starting time of a given part of the task."
        return self.starts[part]

    def get_end(self, part: PART_ID = -1) -> TIME:
        "Get the ending time of of a given part of the task."
        return self.starts[part] + self.durations[part]

    def get_duration(self, part: PART_ID = -1) -> TIME:
        "Get the duration of a given part of the task."
        return self.durations[part]

    def get_assignment(self, part: PART_ID = -1) -> MACHINE_ID:
        "Get the machine assigned to a given part of the task."
        return self.assignments[part]

    def get_start_lb(self, machine: MACHINE_ID = -1) -> TIME:
        "Get the current lower bound for the starting time in a machine."
        if machine != -1:
            return self.start_bounds[machine].lb

        start_lb = MAX_INT
        for machine in self.processing_times:
            machine_start_lb = self.start_bounds[machine].lb
            if machine_start_lb < start_lb:
                start_lb = machine_start_lb

        return start_lb

    def get_start_ub(self, machine: MACHINE_ID = -1) -> TIME:
        "Get the current upper bound for the starting time in a machine."
        if machine != -1:
            return self.start_bounds[machine].ub

        start_ub = 0
        for machine in self.processing_times:
            machine_start_ub = self.start_bounds[machine].ub
            if machine_start_ub > start_ub:
                start_ub = machine_start_ub

        return start_ub

    def get_end_lb(self, machine: MACHINE_ID = -1) -> TIME:
        "Get the current lower bound for the ending time in a machine."
        if machine != -1:
            return self.start_bounds[machine].lb + self._remaining_times[machine]

        end_lb = MAX_INT
        for machine in self.processing_times:
            machine_end_lb = self.start_bounds[machine].lb + self._remaining_times[machine]
            if machine_end_lb < end_lb:
                end_lb = machine_end_lb

        return end_lb

    def get_end_ub(self, machine: MACHINE_ID = -1) -> TIME:
        "Get the current upper bound for the ending time in a machine."
        if machine != -1:
            return self.start_bounds[machine].ub + self._remaining_times[machine]

        end_ub = 0
        for machine in self.processing_times:
            machine_end_ub = self.start_bounds[machine].ub + self._remaining_times[machine]
            if machine_end_ub > end_ub:
                end_ub = machine_end_ub

        return end_ub

    def set_start_lb(self, time: TIME, machine: MACHINE_ID = -1) -> None:
        "Set the lower bound for the starting time in a machine."
        if time < 0: time = 0

        if machine != -1:
            self.start_bounds[machine] = Bounds(
                lb=time,
                ub=self.start_bounds[machine].ub,
            )
            return

        for machine in self.start_bounds:
            self.start_bounds[machine] = Bounds(
                lb=time,
                ub=self.start_bounds[machine].ub,
            )

    def set_start_ub(self, time: TIME, machine: MACHINE_ID = -1) -> None:
        "Set the upper bound for the starting time in a machine."
        if time > MAX_INT: time = MAX_INT

        if machine != -1:
            self.start_bounds[machine] = Bounds(
                lb=self.start_bounds[machine].lb,
                ub=time,
            )
            return

        for machine in self.start_bounds:
            self.start_bounds[machine] = Bounds(
                lb=self.start_bounds[machine].lb,
                ub=time,
            )

    def set_end_lb(self, time: TIME, machine: MACHINE_ID = -1) -> None:
        "Set the lower bound for the ending time in a machine."
        if time < 0: time = 0

        if machine != -1:
            self.start_bounds[machine] = Bounds(
                lb=time - self._remaining_times[machine],
                ub=self.start_bounds[machine].ub,
            )
            return

        for machine in self.start_bounds:
            self.start_bounds[machine] = Bounds(
                lb=time - self._remaining_times[machine],
                ub=self.start_bounds[machine].ub,
            )

    def set_end_ub(self, time: TIME, machine: MACHINE_ID = -1) -> None:
        "Set the upper bound for the ending time in a machine."
        if time > MAX_INT: time = MAX_INT

        if machine != -1:
            self.start_bounds[machine] = Bounds(
                lb=self.start_bounds[machine].lb,
                ub=time - self._remaining_times[machine],
            )
            return

        for machine in self.start_bounds:
            self.start_bounds[machine] = Bounds(
                lb=self.start_bounds[machine].lb,
                ub=time - self._remaining_times[machine],
            )

    def assign(
            self,
            start: TIME,
            machine: MACHINE_ID,
        ) -> None:
        "Assign the execution of the task to a machine at a given start time."
        self.n_parts += 1

        self.starts.append(start)
        self.durations.append(self._remaining_times[machine])
        self.assignments.append(machine)

        self.fixed = True
        for other_machine in self.processing_times:
            bound = start if other_machine == machine else MIN_INT

            self.start_bounds[other_machine] = Bounds(
                lb=bound,
                ub=bound
            )

    def interrupt(self, time: TIME) -> None:
        "Pauses the task's execution at a given time, splitting it into a new part."
        previous_remaining_time = self.durations[-1]
        previsous_start         = self.starts[-1]

        remaining_time     = self.durations[-1] - time + previsous_start
        self.durations[-1] = remaining_time

        for machine in self.processing_times:
            self._remaining_times[machine] = ceil_div(
                self._remaining_times[machine] * remaining_time,
                previous_remaining_time
            )

        self.fixed = False
        self._new_part(time)

    def get_status(self, time: TIME) -> u8:
        "Get the status of the task at a given time."
        # Reverse order because status checkings often occurs in the latest parts

        if not self.fixed:
            if len(self.starts) == 0 or time < self.get_start(0):
                return Status.AWAITING

            return Status.PAUSED

        if time >= self.get_end():
            return Status.COMPLETED

        for part in range(self.n_parts - 1, -1, -1):
            if part > 0 and self.get_end(part - 1) <= time < self.get_start(part):
                return Status.PAUSED

            if self.get_start(part) <= time < self.get_end(part):
                return Status.EXECUTING

        if time < self.get_start():
            return Status.AWAITING

        return Status.UNKNOWN

    def is_available(self, time: TIME, machine: MACHINE_ID = -1) -> bool:
        "Check if the task is available for execution at a given time."
        if machine != -1:
            return self.start_bounds[machine].lb <= time <= self.start_bounds[machine].ub

        for machine in self.start_bounds:
            if self.start_bounds[machine].lb <= time <= self.start_bounds[machine].ub:
                return True

        return False

    def is_awaiting(self) -> bool:
        "Check if the task is currently awaiting execution."
        return not self.fixed

    def is_executing(self, time: TIME, machine: MACHINE_ID = -1) -> bool:
        "Check if the task is being executed at a given time."
        for part in range(self.n_parts - 1, -1, -1):
            if self.get_start(part) <= time < self.get_end(part):
                return machine == -1 or self.get_assignment(part) == machine

        return False

    def is_paused(self, time: TIME) -> bool:
        "Check if the task is paused at a given time."
        for part in range(self.n_parts - 1, 0, -1):
            if self.get_end(part - 1) <= time < self.get_start(part):
                return True

        return False

    def is_completed(self, time: TIME) -> bool:
        "Check if the task is completed at a given time."
        return self.fixed and time >= self.get_end()

    def get_buffer(self, time: TIME) -> str:
        "Get the a string representation of the status of a task at a given time."
        buffer = status_str[self.get_status(time)]

        if buffer == "awaiting" and self.is_available(time):
            buffer = "available"

        return buffer

class Tasks:
    "Container class for the tasks in the scheduling environment."
    n_tasks: int
    n_parts: int
    n_machines: int
    n_jobs: int

    tasks: list[Task]
    jobs: list[list[Task]]

    data     : dict[str, list[Any]]
    jobs_data: dict[str, list[Any]]

    def __init__(
        self,
        data            : dict[str, list[Any]],
        processing_times: list[dict[MACHINE_ID, TIME]],
        jobs_data        : dict[str, list[Any]],
        job_feature     : str = '',
        n_parts         : int = 1,
    ):
        self.n_parts = n_parts

        self.tasks = []
        self.jobs  = []

        self.n_tasks = 0
        machines: set[MACHINE_ID] = set()
        for processing_time in processing_times:
            self.add_task(processing_time)
            machines.update(processing_time.keys())

        self.n_machines = len(machines)

        if not job_feature:
            for alias in JOB_ID_ALIASES:
                if alias in data:
                    job_feature = alias

        job_ids: list[TASK_ID]
        if job_feature in data:
            job_ids = data.pop(job_feature)
            n_jobs  = len(set(job_ids))

        else: # If no job feature is provided, we assume each task is its own job
            job_ids = list(range(self.n_tasks))
            n_jobs  = self.n_tasks

        self.n_jobs = n_jobs

        data['job_id'] = job_ids
        self.jobs = [[] for _ in range(n_jobs)]

        for task_id, job_id in enumerate(job_ids):
            self.jobs[job_id].append(self.tasks[task_id])


        if 'job_id' not in jobs_data:
            if job_feature in jobs_data:
                jobs_data['job_id'] = jobs_data[job_feature]
                del jobs_data[job_feature]

            else:
                jobs_data['job_id'] = list(range(n_jobs))

        self.data      = data
        self.jobs_data = jobs_data


    def add_job(
        self,
        task_ids: list[TASK_ID]
    ) -> None:
        "Add a new job to the tasks container."
        tasks: list[Task] = []
        for task_id in task_ids:
            task = self.tasks[task_id]
            tasks.append(task)

        self.jobs.append(tasks)

    def add_task(
        self,
        processing_times: dict[MACHINE_ID, TIME]
    ) -> None:
        "Add a new task to the tasks container."
        task_id = self.n_tasks
        task = Task(task_id, processing_times)

        self.tasks.append(task)
        self.n_tasks += 1

    def reset(self) -> None:
        "Reset all tasks to their initial state."
        for task in self.tasks:
            task.reset()

    def __len__(self) -> int:
        return self.n_tasks

    def __getitem__(self, task: TASK_ID) -> Task:
        return self.tasks[task]

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    # Getter and setter methods
    def get_task_level_data(self, feature: str) -> list[Any]:
        "Get a specific, task or job, data feature for all tasks."
        if feature in self.data:
            return self.data[feature]

        if feature in self.jobs_data:
            job_data = self.jobs_data[feature]

            return [job_data[job_id] for job_id in self.data['job_id']]

        raise KeyError(f"Feature '{feature}' not found in tasks or jobs data.")

    def get_job_level_data(self, feature: str) -> list[Any]:
        "Get a specific job data feature for all jobs."
        if feature in self.jobs_data:
            return self.jobs_data[feature]

        if feature in self.data:
            if self.n_tasks == self.n_jobs:
                return self.data[feature]

            job_level_data: list[Any] = [None for _ in range(self.n_jobs)]
            for task_id, job_id in enumerate(self.data['job_id']):
                job_level_data[job_id] = self.data[feature][task_id]
            
            return job_level_data
    
        raise KeyError(f"Feature '{feature}' not found in jobs data.")       

    def get_job_tasks(self, job: TASK_ID) -> list[Task]:
        "Get the tasks associated with a specific job."
        return self.jobs[job]

    def get_job_completion(self, job: TASK_ID, time: TIME) -> TIME:
        completion = 0
        for task in self.jobs[job]:
            if task.is_completed(time):
                task_end = task.get_end()
                if task_end > completion:
                    completion = task_end
        
        return completion

    def get_machine_tasks(self, machine: MACHINE_ID) -> list[Task]:
        "Get the tasks that can be processed by a specific machine."
        return [task for task in self.tasks if machine in task.processing_times]

    def get_state(self, current_time: TIME) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        "Get the state of the tasks and jobs at a given time."
        status = [task.get_buffer(current_time) for task in self.tasks]

        task_state: dict[str, list[Any]] = {
            "task_id": list(range(self.n_tasks)),
            **self.data,
            "status": status,
        }

        job_state: dict[str, list[Any]] = {
            "job_id": list(range(len(self.jobs))),
            **self.jobs_data,
        }

        return task_state, job_state

    # TODO: This bound is environment-specific, should be moved to the environment for better estimation
    def tighten_bounds(self, time: int) -> None:
        "Tighten the bounds of the tasks based on the current time."

        max_time = time + sum([
            max(p_times for p_times in task.processing_times.values())
            for task in self.tasks if not task.is_fixed()
        ])

        for task in self.tasks:
            if task.is_fixed():
                continue

            if task.get_start_lb() < time:
                task.set_start_lb(time)

            if task.get_end_ub() > max_time:
                task.set_end_ub(max_time)

    def get_time_ub(self) -> TIME:
        "Get the upper bound for the time in the tasks."
        upper_bound = 0

        for task in self.tasks:
            end_time = task.get_end() if task.is_fixed() else task.get_end_ub()
            if end_time > upper_bound:
                upper_bound = end_time

        return upper_bound

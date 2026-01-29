from typing import Any
from collections.abc import KeysView, Iterator

from dataclasses import dataclass

from cpscheduler.environment._common import MIN_TIME, MAX_TIME, MACHINE_ID, TASK_ID, TIME

GLOBAL_MACHINE_ID: MACHINE_ID = -1


@dataclass(frozen=True)
class TaskHistory:
    assignment: MACHINE_ID
    start_time: TIME
    duration: TIME
    end_time: TIME


class Task:
    task_id: TASK_ID
    job_id: TASK_ID

    history: list[TaskHistory]

    # Static attributes
    # These attributes should only be modified during problem initialization.
    preemptive: bool
    optional: bool
    processing_times: dict[MACHINE_ID, TIME]
    data: dict[str, Any]

    # Constraint Programming variables
    # Do not use these attributes directly anywhere, use the ScheduleState API instead.
    remaining_times_: dict[MACHINE_ID, TIME]
    start_lbs_: dict[MACHINE_ID, TIME]
    start_ubs_: dict[MACHINE_ID, TIME]
    assignment_: MACHINE_ID
    fixed_: bool

    def __init__(self, task_id: TASK_ID, job_id: TASK_ID) -> None:
        self.task_id = task_id
        self.job_id = job_id

        self.history = []

        self.preemptive = False
        self.optional = False
        self.processing_times = {}
        self.data = {}

        self.remaining_times_ = {}
        self.start_lbs_ = {GLOBAL_MACHINE_ID: MAX_TIME}
        self.start_ubs_ = {GLOBAL_MACHINE_ID: MIN_TIME}
        self.assignment_ = GLOBAL_MACHINE_ID
        self.fixed_ = False

    @property
    def machines(self) -> KeysView[MACHINE_ID]:
        "Get the list of machines that can process this task."
        return self.processing_times.keys()

    def __hash__(self) -> int:
        return hash((self.task_id, self.job_id))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Task):
            return NotImplemented

        return (self.task_id == value.task_id) and (self.job_id == value.job_id)

    def __repr__(self) -> str:
        return f"Task(task_id={self.task_id}, job_id={self.job_id})"

    def reset(self) -> None:
        "Resets the task to its initial state."
        self.fixed_ = False

        self.history.clear()

        self.start_lbs_.clear()
        self.start_ubs_.clear()
        self.remaining_times_.clear()
        if self.machines:
            self.start_lbs_[GLOBAL_MACHINE_ID] = 0
            self.start_ubs_[GLOBAL_MACHINE_ID] = MAX_TIME

            for machine in self.machines:
                self.start_lbs_[machine] = 0
                self.start_ubs_[machine] = MAX_TIME
                self.remaining_times_[machine] = self.processing_times[machine]

        else:
            self.start_lbs_[GLOBAL_MACHINE_ID] = MAX_TIME
            self.start_ubs_[GLOBAL_MACHINE_ID] = MIN_TIME

        self.assignment_ = GLOBAL_MACHINE_ID

    # Setter methods
    # These methods are public and only changes static attributes of the task,
    # ONLY during initialization. After a reset call, these methods should not
    # be used anymore.
    def set_processing_time(self, machine: MACHINE_ID, time: TIME) -> None:
        "Set the processing time for a given machine."
        if time < 0:
            return

        self.processing_times[machine] = time

    def set_preemption(self, allow_preemption: bool) -> None:
        "Set whether the task allows preemption."
        self.preemptive = allow_preemption

    def set_optionality(self, optional: bool) -> None:
        "Set whether the task is optional."
        self.optional = optional

    def set_machines(self, machines: list[MACHINE_ID]) -> None:
        "Set the list of machines that can process this task."
        for machine in machines:
            if machine not in self.processing_times:
                raise ValueError(
                    f"Processing time for machine {machine} not set in task {self.task_id}."
                )

        for machine in list(self.processing_times.keys()):
            if machine not in machines:
                del self.processing_times[machine]
                del self.remaining_times_[machine]

    def set_data(self, key: str, value: Any) -> None:
        "Set custom data for the task."
        self.data[key] = value

    # Getter methods
    def get_start_lb(self, machine: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        "Get the current lower bound for the starting time in a machine."
        return self.start_lbs_[machine]

    def get_start_ub(self, machine: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        return self.start_ubs_[machine]

    def get_end_lb(self, machine: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        "Get the current lower bound for the ending time in a machine."
        if machine != -1:
            return self.start_lbs_[machine] + self.remaining_times_[machine]

        end_lb = MAX_TIME
        for machine in self.machines:
            machine_end_lb = self.start_lbs_[machine] + self.remaining_times_[machine]
            if machine_end_lb < end_lb:
                end_lb = machine_end_lb

        return end_lb

    def get_end_ub(self, machine: MACHINE_ID = GLOBAL_MACHINE_ID) -> TIME:
        "Get the current upper bound for the ending time in a machine."
        if machine != -1:
            return self.start_ubs_[machine] + self.remaining_times_[machine]

        end_ub = MIN_TIME
        for machine in self.machines:
            machine_end_ub = self.start_ubs_[machine] + self.remaining_times_[machine]
            if machine_end_ub > end_ub:
                end_ub = machine_end_ub

        return end_ub

    def get_assignment(self) -> MACHINE_ID:
        "Get the machine to which the task is assigned. Returns -1 if unassigned."
        return self.assignment_

    def is_fixed(self) -> bool:
        "Checks if the task has its decision variables fixed_."
        return self.fixed_

    def is_awaiting(self) -> bool:
        "Check if the task is currently awaiting execution."
        return not self.fixed_

    def is_available(self, time: TIME, machine: MACHINE_ID = GLOBAL_MACHINE_ID) -> bool:
        "Check if the task is available for execution at a given time."
        if self.fixed_:
            return False

        return self.start_lbs_[machine] <= time <= self.start_ubs_[machine]

    def is_feasible(self, time: TIME) -> bool:
        "Check if the task is feasible given its current bounds."
        if self.start_lbs_[GLOBAL_MACHINE_ID] > self.start_ubs_[GLOBAL_MACHINE_ID]:
            return False

        if not self.fixed_ and self.start_ubs_[GLOBAL_MACHINE_ID] < time:
            return False

        for machine in self.machines:
            if self.start_lbs_[machine] > self.start_ubs_[machine]:
                return False

        return True


class Job:
    """
    A job is a collection of tasks that are related to each other.
    It serves as a higher-level abstraction for grouping tasks together.
    """

    job_id: TASK_ID
    tasks: list[Task]

    data: dict[str, Any]

    def __init__(self, job_id: TASK_ID) -> None:
        self.job_id = job_id
        self.tasks = []

        self.data = {}

    @property
    def n_tasks(self) -> int:
        "Get the number of tasks in the job."
        return len(self.tasks)

    def __repr__(self) -> str:
        return f"Job(job_id={self.job_id}, n_tasks={self.n_tasks})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Job):
            return False

        return self.job_id == value.job_id

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def add_task(self, task: Task) -> None:
        "Add a task to the job."
        self.tasks.append(task)

    def set_data(self, key: str, value: Any) -> None:
        "Set custom data for the job."
        self.data[key] = value

    def is_available(self, time: TIME) -> bool:
        "Check if any task in the job is available for execution at a given time."
        for task in self.tasks:
            if task.is_available(time):
                return True

        return False

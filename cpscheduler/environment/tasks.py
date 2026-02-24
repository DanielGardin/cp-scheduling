from typing import Any
from collections.abc import KeysView, Iterator

from cpscheduler.environment._common import (
    MAX_TIME,
    MIN_TIME,
    MACHINE_ID,
    TASK_ID,
    TIME,
    GLOBAL_MACHINE_ID,
)


class TaskHistory:
    assignment: MACHINE_ID
    start_time: TIME
    duration: TIME
    end_time: TIME

    def __init__(self, assignment: MACHINE_ID, start_time: TIME, duration: TIME) -> None:
        self.assignment = assignment
        self.start_time = start_time
        self.duration = duration
        self.end_time = start_time + duration

    def __repr__(self) -> str:
        return f"TaskHistory(assignment={self.assignment}, start_time={self.start_time}, duration={self.duration}, end_time={self.end_time})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (self.__class__, (self.assignment, self.start_time, self.duration), ())


class Task:
    task_id: TASK_ID
    job_id: TASK_ID

    preemptive: bool
    optional: bool
    processing_times: dict[MACHINE_ID, TIME]
    data: dict[str, Any]

    def __init__(self, task_id: TASK_ID, job_id: TASK_ID) -> None:
        self.task_id = task_id
        self.job_id = job_id

        self.preemptive = False
        self.optional = False
        self.processing_times = {}
        self.data = {}

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            (self.task_id, self.job_id),
            (
                self.preemptive,
                self.optional,
                self.processing_times,
                self.data,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.preemptive,
            self.optional,
            self.processing_times,
            self.data,
        ) = state

    def __hash__(self) -> int:
        return hash((self.task_id, self.job_id))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Task):
            return NotImplemented

        return (self.task_id == value.task_id) and (self.job_id == value.job_id)

    def __repr__(self) -> str:
        task_repr = f"Task(task_id={self.task_id}, job_id={self.job_id}"

        if self.preemptive:
            task_repr += ", preemptive=True"

        if self.optional:
            task_repr += ", optional=True"

        if self.data:
            for key, value in self.data.items():
                task_repr += f", {key}={value}"

        return task_repr + ")"

    @property
    def machines(self) -> KeysView[MACHINE_ID]:
        "Get the list of machines that can process this task."
        return self.processing_times.keys()

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
            if machine not in self.machines:
                raise ValueError(
                    f"Processing time for machine {machine} not set in task {self.task_id}."
                )

        for machine in list(self.machines):
            if machine not in machines:
                self.processing_times.pop(machine)

    def set_data(self, key: str, value: Any) -> None:
        "Set custom data for the task."
        self.data[key] = value


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

    def __reduce__(self) -> tuple[Any, ...]:
        return (self.__class__, (self.job_id,), (self.tasks, self.data))

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.tasks, self.data = state

    def __repr__(self) -> str:
        return f"Job(job_id={self.job_id}, n_tasks={self.n_tasks})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Job):
            return False

        return self.job_id == value.job_id

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    @property
    def n_tasks(self) -> int:
        "Get the number of tasks in the job."
        return len(self.tasks)

    @property
    def task_ids(self) -> list[TASK_ID]:
        "Get the list of task IDs in the job."
        return [task.task_id for task in self.tasks]

    def add_task(self, task: Task) -> None:
        "Add a task to the job."
        self.tasks.append(task)

    def set_data(self, key: str, value: Any) -> None:
        "Set custom data for the job."
        self.data[key] = value


def initialize_matrix(n_rows: int, n_cols: int, value: TIME) -> list[TIME]:
    return [value] * (n_rows * n_cols)


class Bounds:
    __slots__ = ["n_machines", "lbs", "ubs", "global_lbs", "global_ubs"]

    n_machines: int

    lbs: list[TIME]
    ubs: list[TIME]
    global_lbs: list[TIME]
    global_ubs: list[TIME]

    min_lb: TIME
    max_ub: TIME

    def __init__(self, tasks: list[Task], n_machines: int) -> None:
        self.n_machines = n_machines

        self.lbs = initialize_matrix(len(tasks), n_machines, MAX_TIME)
        self.ubs = initialize_matrix(len(tasks), n_machines, MIN_TIME)

        for task_id, task in enumerate(tasks):
            for machine in task.machines:
                self.lbs[task_id * self.n_machines + machine] = MIN_TIME
                self.ubs[task_id * self.n_machines + machine] = MAX_TIME

        self.global_lbs = [MIN_TIME for _ in tasks]
        self.global_ubs = [MAX_TIME for _ in tasks]

        self.recompute_all_global_bounds()

    def recompute_global_bounds(self, task_id: TASK_ID) -> None:
        start = task_id * self.n_machines
        end = start + self.n_machines

        # self.global_lbs[task_id] = min(self.lbs[start:end])
        # self.global_ubs[task_id] = max(self.ubs[start:end])

        min_lb = self.lbs[start]
        max_ub = self.ubs[start]

        for i in range(start + 1, end):
            if self.lbs[i] < min_lb:
                min_lb = self.lbs[i]

            if self.ubs[i] > max_ub:
                max_ub = self.ubs[i]

        self.global_lbs[task_id] = min_lb
        self.global_ubs[task_id] = max_ub

    def recompute_all_global_bounds(self) -> None:
        for task_id in range(len(self.lbs) // self.n_machines):
            start = task_id * self.n_machines
            end = start + self.n_machines
            self.global_lbs[task_id] = min(self.lbs[start:end])
            self.global_ubs[task_id] = max(self.ubs[start:end])

    def __repr__(self) -> str:
        return f"Bounds(lbs={self.lbs}, ubs={self.ubs}, global_lbs={self.global_lbs}, global_ubs={self.global_ubs})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            ([], self.n_machines),  # Dummy arguments for __init__
            (self.lbs, self.ubs, self.global_lbs, self.global_ubs),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.lbs, self.ubs, self.global_lbs, self.global_ubs = state

    def get_lb(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> TIME:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_lbs[task_id]

        else:
            return self.lbs[task_id * self.n_machines + machine_id]

    def get_ub(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> TIME:
        if machine_id == GLOBAL_MACHINE_ID:
            return self.global_ubs[task_id]
        else:
            return self.ubs[task_id * self.n_machines + machine_id]


class ScheduleVariables:
    __slots__ = [
        "remaining_times",
        "assignment",
        "locked",
        "present",
        "start",
        "end",
        "n_machines_feasible"
    ]

    remaining_times: list[TIME]
    assignment: list[MACHINE_ID]

    locked: list[bool]
    present: list[bool]

    start: Bounds
    end: Bounds
    n_machines_feasible: list[int]

    def __init__(self, tasks: list[Task], n_machines: int) -> None:
        self.remaining_times = initialize_matrix(len(tasks), n_machines, MAX_TIME)
        self.assignment = [GLOBAL_MACHINE_ID for _ in tasks]

        self.locked = [False for _ in tasks]
        self.present = [not task.optional for task in tasks]

        self.start = Bounds(tasks, n_machines)
        self.end = Bounds(tasks, n_machines)

        self.n_machines_feasible = [0] * len(tasks)

        for task_id, task in enumerate(tasks):
            for machine, processing_time in task.processing_times.items():
                idx = task_id * n_machines + machine

                self.remaining_times[idx] = processing_time
                self.start.ubs[idx] = self.end.ubs[idx] - processing_time
                self.end.lbs[idx] = self.start.lbs[idx] + processing_time

            self.start.recompute_global_bounds(task_id)
            self.end.recompute_global_bounds(task_id)

    def __repr__(self) -> str:
        return f"ScheduleVariables(remaining_times={self.remaining_times}, assignment={self.assignment}, locked={self.locked}, present={self.present}, start={self.start}, end={self.end})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            ([], self.start.n_machines),  # Dummy argument for __init__
            (
                self.remaining_times,
                self.assignment,
                self.locked,
                self.present,
                self.start,
                self.end,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.remaining_times,
            self.assignment,
            self.locked,
            self.present,
            self.start,
            self.end,
        ) = state

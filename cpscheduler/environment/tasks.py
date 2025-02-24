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

from typing import Any, Iterator

from textwrap import dedent
from enum import Enum

from .utils import MAX_INT

class Status(Enum):
    AWAITING  = 0  # time < start_lb[0] or waiting for a machine
    EXECUTING = 1  # start_lb[i] <               = time < start_lb[i] + duration[i] for some i
    PAUSED    = 2  # start_lb[i] + duration[i] < = time < start_lb[i+1] for some i
    COMPLETED = 3  # time >                      = start_lb[-1] + duration[-1]
    UNKNOWN   = 4  # unknown status


status_str = {
    Status.AWAITING : "awaiting",
    Status.EXECUTING: "executing",
    Status.PAUSED   : "paused",
    Status.COMPLETED: "completed",
    Status.UNKNOWN  : "unknown",
}


class Task:
    start_lbs  : list[int]
    start_ubs  : list[int]
    durations  : list[int]
    assignments: list[int]

    def __init__(
        self,
        task_id: int,
        processing_time: int,
    ) -> None:
        self.task_id = task_id
        self.processing_time = processing_time

        self.start_lbs   = []
        self.start_ubs   = []
        self.durations   = []
        self.assignments = []

        self.remaining_time = processing_time
        self.status = Status.AWAITING
        self.n_parts = 0


    def reset(self) -> None:
        self.remaining_time = self.processing_time
        self.status = Status.AWAITING

        self.n_parts = 0

        self.start_lbs.clear()
        self.start_ubs.clear()
        self.durations.clear()
        self.assignments.clear()

        self._new_part(0)

    def _new_part(self, time: int) -> None:
        self.start_lbs.append(time)
        self.start_ubs.append(MAX_INT)
        self.durations.append(MAX_INT)
        self.assignments.append(-1)

        self.n_parts += 1

    def is_fixed(self, part: int = -1) -> bool:
        return self.start_lbs[part] == self.start_ubs[part]

    def get_start(self, part: int = 0) -> int:
        return self.start_lbs[part] if self.is_fixed(part) else MAX_INT

    def get_end(self, part: int = -1) -> int:
        return min(self.start_lbs[part] + self.durations[part], MAX_INT)

    def get_start_lb(self) -> int:
        return self.start_lbs[-1]

    def get_start_ub(self) -> int:
        return self.start_ubs[-1]

    def get_end_lb(self) -> int:
        return min(self.start_lbs[-1] + self.remaining_time, MAX_INT)

    def get_end_ub(self) -> int:
        return min(self.start_ubs[-1] + self.remaining_time, MAX_INT)

    def set_start_lb(self, time: int) -> None:
        self.start_lbs[-1] = time

    def set_start_ub(self, time: int) -> None:
        self.start_ubs[-1] = time

    def set_end_lb(self, time: int) -> None:
        self.start_lbs[-1] = time - self.remaining_time

    def set_end_ub(self, time: int) -> None:
        self.start_ubs[-1] = time - self.remaining_time

    def execute(self, time: int, machine: int) -> None:
        self.start_lbs[-1] = time
        self.start_ubs[-1] = time
        self.assignments[-1] = machine
        self.durations[-1] = self.remaining_time

    def pause(self, time: int) -> None:
        duration = min(self.remaining_time, time - self.start_lbs[-1])
        self.durations[-1] = duration

        self.remaining_time -= duration

        if self.remaining_time > 0:
            self._new_part(time)

    def get_status(self, time: int) -> Status:
        if time < self.start_lbs[0]:
            return Status.AWAITING

        if time >= self.start_lbs[-1] + self.durations[-1]:
            return Status.COMPLETED

        # Reverse order because status checkings often occurs in the latest parts
        for part in range(self.n_parts - 1, -1, -1):
            if part > 0 and self.get_end(part - 1) <= time < self.get_start(part):
                return Status.PAUSED

            if self.start_lbs[part] <= time < self.start_ubs[part]:
                return Status.AWAITING

            if self.get_start(part) <= time < self.get_end(part):
                return Status.EXECUTING

        return Status.UNKNOWN

    def is_available(self, time: int) -> bool:
        return self.start_lbs[-1] <= time < self.start_ubs[-1]

    def is_awaiting(self, time: int) -> bool:
        return time < self.start_lbs[0] or self.is_available(time)

    def is_executing(self, time: int) -> bool:
        for part in range(self.n_parts - 1, -1, -1):
            if self.get_start(part) <= time < self.get_end(part):
                return True

        return False

    def is_paused(self, time: int) -> bool:
        for part in range(self.n_parts - 1, 0, -1):
            if self.get_end(part - 1) <= time < self.get_start(part):
                return True

        return False

    def is_completed(self, time: int) -> bool:
        return time >= self.start_lbs[-1] + self.durations[-1]

    def get_buffer(self, time: int) -> str:
        buffer = status_str[self.get_status(time)]

        if buffer == "awaiting" and self.is_available(time):
            buffer = "available"

        return buffer


class Tasks:
    n_tasks: int
    n_parts: int

    data: dict[str, list[Any]]
    tasks: list[Task]
    jobs: dict[int, list[Task]]

    def __init__(
        self,
        data: dict[str, list[Any]],
        n_parts: int = 1,  # if we allow preemption, we must define a maximum number of splits
    ):
        self.n_parts = n_parts
        self.tasks = []

        self.jobs = {}

        self.n_tasks = 0
        self.data = data

    def add_task(
        self,
        processing_time: int,
        job: int,
    ) -> None:
        task_id = self.n_tasks
        task = Task(
            task_id,
            processing_time,
        )

        self.tasks.append(task)
        if job not in self.jobs:
            self.jobs[job] = []

        self.jobs[job].append(task)

        self.n_tasks += 1

    def reset(self) -> None:
        for task in self.tasks:
            task.reset()

    def __len__(self) -> int:
        return self.n_tasks

    def __getitem__(self, task: int) -> Task:
        return self.tasks[task]

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    # Getter and setter methods
    def get_feature(self, task: int, feature: str) -> Any:
        return self.data[feature][task]

    def get_job_tasks(self, job: int) -> list[Task]:
        return self.jobs[job]

    def get_state(self, current_time: int) -> dict[str, list[Any]]:
        status = [task.get_buffer(current_time) for task in self.tasks]

        return {
            "task_id": list(range(self.n_tasks)),
            **self.data,
            "remaining_time": [task.remaining_time for task in self.tasks],
            "status": status,
        }

    def get_time_ub(self) -> int:
        last_upper_bound = max(task.get_end_ub() for task in self.tasks)

        return last_upper_bound

    def export_model(
        self,
    ) -> str:
        model = """\
            include "globals.mzn";

            int: horizon;
            int: num_tasks;
            int: num_parts;

            array[1..num_tasks] of int: processing_time;
            array[1..num_tasks] of int: start_lb;
            array[1..num_tasks] of int: start_ub;

            array[1..num_tasks, 1..num_parts] of var 0..horizon: start;
            array[1..num_tasks, 1..num_parts] of var 0..horizon: duration;
            array[1..num_tasks, 1..num_parts] of var 0..horizon: end;

            constraint forall(t in 1..num_tasks) (
                start_lb[t] <= start[t, 1] /\\ start[t, 1] <= start_ub[t]
            );

            constraint forall(t in 1..num_tasks, p in 1..num_parts) (
                start[t, p] + duration[t, p] = end[t, p]
            );
        """

        if self.n_parts > 1:
            model += """\
            constraint forall(t in 1..num_tasks, p in 2..num_parts)(
                end[t, p-1] < start[t, p]
            );

            constraint forall(t in 1..num_tasks, p in 2..num_parts)(
                (duration[t, p-1] = 0) -> (duration[t, p] = 0)
            );
            """

        return dedent(model)

    def export_data(
        self,
    ) -> str:
        data = f"""
            horizon = {self.get_time_ub()};
            num_tasks = {self.n_tasks};
            num_parts = {self.n_parts};

            processing_time = [{', '.join([str(task.processing_time) for task in self.tasks])}];
            start_lb  = [{', '.join([str(task.get_start_lb()) for task in self.tasks])}];
            start_ub  = [{', '.join([str(task.get_start_ub()) for task in self.tasks])}];
        """

        return dedent(data)

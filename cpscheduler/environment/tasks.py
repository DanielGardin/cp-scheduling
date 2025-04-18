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

from typing import Any, Iterator, Optional, NamedTuple, ClassVar

from textwrap import dedent

from .common import MAX_INT, MIN_INT

class Status:
    AWAITING : ClassVar[int] = 0  # time < start_lb[0] or waiting for a machine
    EXECUTING: ClassVar[int] = 1  # start_lb[i] <               = time < start_lb[i] + duration[i] for some i
    PAUSED   : ClassVar[int] = 2  # start_lb[i] + duration[i] < = time < start_lb[i+1] for some i
    COMPLETED: ClassVar[int] = 3  # time >                      = start_lb[-1] + duration[-1]
    UNKNOWN  : ClassVar[int] = 4  # unknown status


status_str = {
    Status.AWAITING : "awaiting",
    Status.EXECUTING: "executing",
    Status.PAUSED   : "paused",
    Status.COMPLETED: "completed",
    Status.UNKNOWN  : "unknown",
}

def ceil_div(a: int, b: int) -> int:
    return -(-a // b)

class Bounds(NamedTuple):
    lb: int = 0
    ub: int = MAX_INT


class Task:
    processing_times: dict[int, int]

    starts     : list[int]
    durations  : list[int]
    assignments: list[int]

    start_bounds: dict[int, Bounds]

    def __init__(
        self,
        task_id: int,
        processing_times: dict[int, int],
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
        return ""


    def reset(self) -> None:
        self._remaining_time = self.processing_times.copy()
    
        self.fixed   = False
        self.n_parts = 0

        self.starts.clear()
        self.durations.clear()
        self.assignments.clear()

        self._new_part(0)

    def _new_part(self, time: int) -> None:
        for machine in self.processing_times:
            self.start_bounds[machine] = Bounds(
                lb=time,
            )

    def is_fixed(self) -> bool:
        return self.fixed

    def get_start(self, part: int = 0) -> int:
        return self.starts[part]

    def get_end(self, part: int = -1) -> int:
        return min(self.starts[part] + self.durations[part], MAX_INT)

    def get_duration(self, part: int = -1) -> int:
        return self.durations[part]

    def get_assignment(self, part: int = -1) -> int:
        return self.assignments[part]

    def get_start_lb(self, machine: int = -1) -> int:
        if machine != -1:
            return self.start_bounds[machine].lb

        return min(self.start_bounds[machine].lb for machine in self.processing_times)

    def get_start_ub(self, machine: int = -1) -> int:
        if machine != -1:
            return self.start_bounds[machine].ub

        return min(self.start_bounds[machine].ub for machine in self.processing_times)

    def get_end_lb(self, machine: int = -1) -> int:
        if machine != -1:
            return self.start_bounds[machine].lb + self._remaining_times[machine]

        return min(
            self.start_bounds[machine].lb + self._remaining_times[machine]
            for machine in self.processing_times
        )

    def get_end_ub(self, machine: int = -1) -> int:
        if machine != -1:
            return self.start_bounds[machine].ub + self._remaining_times[machine]

        return min(
            self.start_bounds[machine].ub + self._remaining_times[machine]
            for machine in self.processing_times
        )

    def set_start_lb(self, time: int, machine: int = -1) -> None:
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

    def set_start_ub(self, time: int, machine: int = -1) -> None:
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

    def set_end_lb(self, time: int, machine: int = -1) -> None:
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


    def set_end_ub(self, time: int, machine: int = -1) -> None:
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

    def get_remaining_time(self, time: int) -> int:
        return max(0, min((self.durations[-1] - time + self.starts[-1]), self.durations[-1]))


    def assign(
            self,
            start: int,
            machine: int,
        ) -> None:
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


    def interrupt(self, time: int) -> None:
        if self.n_parts == 0:
            raise ValueError(
                f"Task {self.task_id} has not been started yet. Cannot interrupt."
            )

        previous_time  = self.durations[-1]
        remaining_time = self.get_remaining_time(time)

        if remaining_time == 0:
            return

        self.durations[-1] -= remaining_time

        for machine in self.processing_times:
            self._remaining_times[machine] = ceil_div(
                remaining_time * self.processing_times[machine],
                previous_time
            )

        self.fixed = False
        self._new_part(time)

    def get_status(self, time: int) -> int:
        # Reverse order because status checkings often occurs in the latest parts
        if not self.fixed:
            if len(self.starts) == 0 or time < self.get_start(0):
                return Status.AWAITING

            return Status.PAUSED

        for part in range(self.n_parts - 1, -1, -1):
            if part > 0 and self.get_end(part - 1) <= time < self.get_start(part):
                return Status.PAUSED

            if self.get_start(part) <= time < self.get_end(part):
                return Status.EXECUTING
        
        # If it's not paused, but also it's not fixed, it must be awaiting
        if not self.fixed:
            return Status.AWAITING

        if time >= self.get_end():
            return Status.COMPLETED

        return Status.UNKNOWN

    def is_available(self, time: int, machine: int = -1) -> bool:
        if machine != -1:
            return self.start_bounds[machine].lb <= time < self.start_bounds[machine].ub

        for machine in self.start_bounds:
            if self.start_bounds[machine].lb <= time < self.start_bounds[machine].ub:
                return True

        return False

    def is_awaiting(self) -> bool:
        return not self.fixed

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
        return self.fixed and time >= self.get_end()

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
        data: Optional[dict[str, list[Any]]] = None,
        n_parts: int = 1,  # if we allow preemption, we must define a maximum number of splits
    ):
        self.n_parts = n_parts
        self.tasks = []

        self.jobs = {}

        self.n_tasks = 0

        self.data = data if data is not None else {}

        self.n_expected_tasks = 0
        for key, features in self.data.items():
            if self.n_expected_tasks == 0 and len(features) > 0:
                self.n_expected_tasks = len(features)
            
            if len(features) != self.n_expected_tasks:
                raise ValueError(
                    f"Feature {key} has {len(features)} tasks, expected {self.n_expected_tasks}"
                )


    def handle_data(self, data: dict[str, Any]) -> None:
        if self.n_expected_tasks > self.n_tasks:
            raise ValueError(
                f"Unexpected new data, {self.n_expected_tasks - self.n_parts} tasks waiting to be loaded."
            )
    
        elif self.n_expected_tasks == self.n_tasks:
            self.n_expected_tasks += 1

        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []

            self.data[key].append(value)

    def add_task(
        self,
        processing_times: dict[int, int],
        job: int,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        task_id = self.n_tasks
        task = Task(task_id, processing_times)

        if job not in self.jobs:
            self.jobs[job] = []

        self.tasks.append(task)
        self.jobs[job].append(task)

        if data is not None:
            self.handle_data(data)

        elif self.n_expected_tasks <= self.n_tasks:
            raise ValueError(
                f"Expected new data from this task. Check if the data is correctly loaded."
            )

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
            "status": status,
        }

    def get_time_ub(self) -> int:
        last_upper_bound = max(task.get_end_ub() for task in self.tasks)

        return last_upper_bound


    # TODO: Check if function end is better than array end.
    # TODO: Add search annotations and warm up.
    def export_model(
        self,
    ) -> str:
        model = """\
            include "globals.mzn";

            int: horizon;
            int: num_tasks;
            int: num_parts;
            int: num_jobs;

            array[1..num_tasks] of 0..horizon:  start_lb;
            array[1..num_tasks] of 0..horizon:  start_ub;
            array[1..num_jobs] of set of 1..num_tasks: job;

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
        data = dedent(f"""
            horizon = {self.get_time_ub()};
            num_tasks = {self.n_tasks};
            num_parts = {self.n_parts};
            num_jobs = {len(self.jobs)};

            start_lb  = [{', '.join([str(task.get_start_lb()) for task in self.tasks])}];
            start_ub  = [{', '.join([str(task.get_start_ub()) for task in self.tasks])}];
        """)

        data += "job = [\n"
        for job_tasks in self.jobs.values():
            data += "    {" + ", ".join([str(task.task_id+1) for task in job_tasks]) + "},\n"           
        data += "];\n"

        return data

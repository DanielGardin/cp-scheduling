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

from typing import Any
from typing_extensions import Self

from mypy_extensions import u8

from cpscheduler.environment._common import (
    MIN_INT,
    MAX_INT,
    MACHINE_ID,
    TASK_ID,
    PART_ID,
    TIME,
    Status,
    ceil_div,
)


class Bounds:
    "Store the lower and upper bounds for decision variables."

    def __init__(self, lb: TIME = 0, ub: TIME = MAX_INT) -> None:
        self.lb = lb
        self.ub = ub

    def __reduce__(self) -> tuple[type, tuple[TIME, TIME]]:
        return (self.__class__, (self.lb, self.ub))

    def __setstate__(self, state: tuple[TIME, TIME]) -> None:
        self.lb, self.ub = state

    def reset(self) -> None:
        "Resets the bounds to their initial state."
        self.lb = 0
        self.ub = MAX_INT

    def set(self, lb: TIME = 0, ub: TIME = MAX_INT) -> None:
        self.lb = lb
        self.ub = ub

    def fix(self, time: TIME) -> None:
        self.lb = time
        self.ub = time

    def nullify(self) -> None:
        "Set the bounds to null values."
        self.lb = MAX_INT
        self.ub = MIN_INT

    @classmethod
    def null(cls) -> Self:
        "Create a null bounds object."
        return cls(lb=MAX_INT, ub=MIN_INT)

    def is_feasible(self) -> bool:
        "Check if the bounds are feasible."
        return self.lb <= self.ub

    def __repr__(self) -> str:
        return f"Bounds(lb={self.lb}, ub={self.ub})"


# There are several bugs hidden in the implementation of `global_bound`
class Task:
    """
    Minimal unit of work that can be scheduled. The task does not know anything about the
    environment, it only knows about its own state and the processing times for each machine.

    The task is defined by:
        - task_id: unique identifier for the task, usually its position in the list of tasks
        - processing_times: time required to complete the task

    The environment has complete information about every task and orchestrates the scheduling
    process by modifying the task state.
    The task can be split into multiple parts, each with its own starting time and duration.
    """

    task_id: TASK_ID
    job_id: TASK_ID
    n_parts: PART_ID

    preemptive: bool

    starts: list[TIME]
    durations: list[TIME]
    assignments: list[MACHINE_ID]

    processing_times: dict[MACHINE_ID, TIME]
    machines: list[MACHINE_ID]

    remaining_times: dict[MACHINE_ID, TIME]
    start_bounds: dict[MACHINE_ID, Bounds]
    global_bound: Bounds

    fixed: bool

    def __init__(self, task_id: TASK_ID, job_id: TASK_ID) -> None:
        self.task_id = task_id
        self.job_id = job_id
        self.n_parts = 0

        self.preemptive = False

        self.starts = []
        self.durations = []
        self.assignments = []

        self.processing_times = {}
        self.machines = []

        self.remaining_times = {}
        self.global_bound = Bounds()
        self.start_bounds = {}

        self.fixed = False

    def __hash__(self) -> int:
        return hash((self.task_id, self.job_id))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Task):
            return NotImplemented

        return (self.task_id == value.task_id) and (self.job_id == value.job_id)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.task_id, self.job_id),
            (
                self.n_parts,
                self.starts,
                self.durations,
                self.assignments,
                self.processing_times,
                self.machines,
                self.remaining_times,
                self.start_bounds,
                self.global_bound,
                self.fixed,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.n_parts,
            self.starts,
            self.durations,
            self.assignments,
            self.processing_times,
            self.machines,
            self.remaining_times,
            self.start_bounds,
            self.global_bound,
            self.fixed,
        ) = state

    def __repr__(self) -> str:
        representation = f"Task(id={self.task_id}"

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
        self.n_parts = 0

        self.starts.clear()
        self.durations.clear()
        self.assignments.clear()

        for machine in self.machines:
            self.remaining_times[machine] = self.processing_times[machine]
            self.start_bounds[machine].reset()

        self.global_bound.reset()
        self.fixed = False

    def is_fixed(self) -> bool:
        "Checks if the task has its decision variables fixed."
        return self.fixed

    def is_feasible(self, time: TIME) -> bool:
        "Check if the task is feasible given its current bounds."
        return self.global_bound.is_feasible() and time <= self.global_bound.ub

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

    def get_processed_time(self, time: TIME, machine: MACHINE_ID = -1) -> TIME:
        """
        Get the time processed by the task at a given time.
        If machine is specified, return the time processed by that machine.
        """
        processed_time = 0
        if machine != -1:
            if machine not in self.remaining_times:
                return processed_time

            for part in range(self.n_parts):
                if self.get_assignment(part) == machine:
                    if self.get_start(part) <= time < self.get_end(part):
                        processed_time += time - self.get_start(part)

                    if time >= self.get_end(part):
                        processed_time += self.get_duration(part)

            return processed_time

        for part in range(self.n_parts):
            if self.get_start(part) <= time < self.get_end(part):
                processed_time += time - self.get_start(part)

            if time >= self.get_end(part):
                processed_time += self.get_duration(part)

        return processed_time

    def get_start_lb(self, machine: MACHINE_ID = -1) -> TIME:
        "Get the current lower bound for the starting time in a machine."
        return self.global_bound.lb if machine == -1 else self.start_bounds[machine].lb

    def get_start_ub(self, machine: MACHINE_ID = -1) -> TIME:
        return self.global_bound.ub if machine == -1 else self.start_bounds[machine].ub

    def get_end_lb(self, machine: MACHINE_ID = -1) -> TIME:
        "Get the current lower bound for the ending time in a machine."
        if machine != -1:
            return self.start_bounds[machine].lb + self.remaining_times[machine]

        end_lb = MAX_INT
        for machine in self.machines:
            machine_end_lb = (
                self.start_bounds[machine].lb + self.remaining_times[machine]
            )
            if machine_end_lb < end_lb:
                end_lb = machine_end_lb

        return end_lb

    def get_end_ub(self, machine: MACHINE_ID = -1) -> TIME:
        "Get the current upper bound for the ending time in a machine."
        if machine != -1:
            return self.start_bounds[machine].ub + self.remaining_times[machine]

        end_ub = 0
        for machine in self.machines:
            machine_end_ub = (
                self.start_bounds[machine].ub + self.remaining_times[machine]
            )
            if machine_end_ub > end_ub:
                end_ub = machine_end_ub

        return end_ub

    def set_start_lb(self, time: TIME, machine: MACHINE_ID = -1) -> None:
        "Set the lower bound for the starting time in a machine."
        if time < 0:
            time = 0

        if machine != -1:
            self.start_bounds[machine].lb = time
            self.global_bound.lb = min(bound.lb for bound in self.start_bounds.values())

        else:
            self.global_bound.lb = time
            for start_bound in self.start_bounds.values():
                start_bound.lb = time

    def set_start_ub(self, time: TIME, machine: MACHINE_ID = -1) -> None:
        "Set the upper bound for the starting time in a machine."
        if time > MAX_INT:
            time = MAX_INT

        if machine != -1:
            self.start_bounds[machine].ub = time
            self.global_bound.ub = max(bound.ub for bound in self.start_bounds.values())

        else:
            self.global_bound.ub = time
            for start_bound in self.start_bounds.values():
                start_bound.ub = time

    def set_end_lb(self, time: TIME, machine: MACHINE_ID = -1) -> None:
        "Set the lower bound for the ending time in a machine."
        if time < 0:
            time = 0

        if machine != -1:
            self.start_bounds[machine].lb = time - self.remaining_times[machine]

        else:
            for machine in self.machines:
                self.start_bounds[machine].lb = time - self.remaining_times[machine]

        self.global_bound.lb = min(bound.lb for bound in self.start_bounds.values())

    def set_end_ub(self, time: TIME, machine: MACHINE_ID = -1) -> None:
        "Set the upper bound for the ending time in a machine."
        if time > MAX_INT:
            time = MAX_INT

        if machine != -1:
            self.start_bounds[machine].ub = time - self.remaining_times[machine]

        else:
            for machine in self.machines:
                self.start_bounds[machine].ub = time - self.remaining_times[machine]

        self.global_bound.ub = max(bound.ub for bound in self.start_bounds.values())

    def set_processing_time(self, machine: MACHINE_ID, time: TIME) -> None:
        "Set the processing time for a given machine."
        if time < 0:
            return

        self.processing_times[machine] = time
        self.remaining_times[machine] = time
        self.machines.append(machine)
        self.start_bounds[machine] = Bounds()

    def set_preemption(self, allow_preemption: bool) -> None:
        "Set whether the task allows preemption."
        self.preemptive = allow_preemption

    def execute(
        self,
        time: TIME,
        machine: MACHINE_ID = -1,
    ) -> bool:
        "Assign the execution of the task to a machine at a given start time. Returns True if successful."
        if machine == -1:
            for machine in self.machines:
                if self.is_available(time, machine):
                    break

            else:
                return False

        elif machine not in self.processing_times or not self.is_available(
            time, machine
        ):
            return False

        self.n_parts += 1

        self.starts.append(time)
        self.durations.append(self.remaining_times[machine])
        self.assignments.append(machine)

        self.fixed = True
        self.global_bound.fix(time)
        for other_machine in self.machines:
            if other_machine == machine:
                self.start_bounds[other_machine].fix(time)

            else:
                self.start_bounds[other_machine].nullify()

        return True

    def pause(self, time: TIME) -> bool:
        "Pauses the task's execution at a given time, splitting it into a new part."
        if not self.fixed or not self.preemptive:
            return False

        prev_duration = self.durations[-1]
        actual_duration = time - self.starts[-1]

        remaining_time = prev_duration - actual_duration
        if remaining_time <= 0:
            return False

        self.durations[-1] = actual_duration

        self.global_bound.set(time, MAX_INT)
        for machine in self.machines:
            self.remaining_times[machine] = ceil_div(
                self.remaining_times[machine] * remaining_time, prev_duration
            )

            self.start_bounds[machine].set(time, MAX_INT)

        self.fixed = False

        return True

    def get_status(self, time: TIME) -> u8:
        "Get the status of the task at a given time."
        # Reverse order because status checkings often occurs in the latest parts

        if not self.fixed:
            if not self.is_feasible(time):
                return Status.UNFEASIBLE

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

        raise RuntimeError(f"Inconsistent task state detected for task {self.task_id}.")

    def is_available(self, time: TIME, machine: MACHINE_ID = -1) -> bool:
        "Check if the task is available for execution at a given time."
        if self.fixed:
            return False

        if machine != -1:
            return (
                self.start_bounds[machine].lb <= time <= self.start_bounds[machine].ub
            )

        return self.global_bound.lb <= time <= self.global_bound.ub

    def is_awaiting(self) -> bool:
        "Check if the task is currently awaiting execution."
        return not self.fixed

    def is_executing(self, time: TIME, machine: MACHINE_ID = -1) -> bool:
        "Check if the task is being executed at a given time."
        for part in range(self.n_parts - 1, -1, -1):
            if self.get_end(part) <= time:
                break

            if self.get_start(part) <= time:
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

from typing import Any
from collections.abc import KeysView

from cpscheduler.environment.constants import (
    MAX_TIME,
    MIN_TIME,
    MACHINE_ID,
    TASK_ID,
    TIME,
    GLOBAL_MACHINE_ID,
    PresenceType,
    Presence,
)

from cpscheduler.utils.list_utils import convert_to_list


def check_instance_consistency(instance: dict[str, list[Any]]) -> int:
    "Check if all lists in the instance have the same length."
    lengths = {len(v) for v in instance.values()}

    if len(lengths) > 1:
        raise ValueError(
            "Inconsistent instance data: all lists must have the same length."
        )

    return lengths.pop() if lengths else 0


class ProblemInstance:
    __slots__ = (
        "job_ids",
        "job_tasks",
        "preemptive",
        "optional",
        "processing_times",
        "task_instance",
        "n_tasks",
        "n_jobs",
        "_n_machines",
    )

    job_ids: list[TASK_ID]
    job_tasks: list[list[TASK_ID]]

    preemptive: list[bool]
    optional: list[bool]
    processing_times: list[dict[MACHINE_ID, TIME]]

    task_instance: dict[str, list[Any]]

    n_tasks: int
    n_jobs: int
    _n_machines: int

    def __init__(
        self,
        task_instance: dict[str, list[Any]],
    ) -> None:
        self.task_instance = task_instance.copy()

        self.n_tasks = check_instance_consistency(task_instance)
        assert self.n_tasks > 0, "Instance must contain at least one task."

        job_ids = convert_to_list(
            (
                task_instance["job"]
                if "job" in task_instance
                else range(self.n_tasks)
            ),
            TASK_ID,
        )

        self.job_ids = job_ids
        self.n_jobs = max(job_ids) + 1 if job_ids else 0

        self.preemptive = [False] * self.n_tasks
        self.optional = [False] * self.n_tasks
        self.processing_times = [{} for _ in range(self.n_tasks)]

        self._n_machines = 0

        self.job_tasks = [[] for _ in range(self.n_jobs)]
        for task_id, job_id in enumerate(job_ids):
            self.job_tasks[job_id].append(task_id)

        self.task_instance["job_id"] = self.job_ids
        self.task_instance["task_id"] = list(range(self.n_tasks))

    @property
    def n_machines(self) -> int:
        if self._n_machines > 0:
            return self._n_machines

        for p_times in self.processing_times:
            for machine_id in p_times:
                if machine_id >= self._n_machines:
                    self._n_machines = machine_id + 1

        return self._n_machines

    def is_preemptive(self, task_id: TASK_ID) -> bool:
        "Check if a task allows preemption."
        return self.preemptive[task_id]

    def is_optional(self, task_id: TASK_ID) -> bool:
        "Check if a task is optional."
        return self.optional[task_id]

    def get_processing_time(
        self, task_id: TASK_ID, machine_id: MACHINE_ID
    ) -> TIME:
        "Get the processing time for a given task and machine."
        return self.processing_times[task_id].get(machine_id, MAX_TIME)

    def get_machines(self, task_id: TASK_ID) -> KeysView[MACHINE_ID]:
        "Get the set of machines that can process a given task."
        return self.processing_times[task_id].keys()

    def set_preemption(
        self, task_id: TASK_ID, allow_preemption: bool = True
    ) -> None:
        "Set whether a task allows preemption."
        self.preemptive[task_id] = allow_preemption

    def set_optionality(self, task_id: TASK_ID, optional: bool = True) -> None:
        "Set whether a task is optional."
        self.optional[task_id] = optional

    def set_processing_time(
        self, task_id: TASK_ID, machine_id: MACHINE_ID, time: TIME
    ) -> None:
        "Set the processing time for a given task and machine."
        if time < 0:
            raise ValueError("Processing time cannot be negative.")

        self.processing_times[task_id][machine_id] = time

    def remove_machine(self, task_id: TASK_ID, machine_id: MACHINE_ID) -> None:
        "Remove a machine from processing a given task."
        if machine_id in self.processing_times[task_id]:
            del self.processing_times[task_id][machine_id]

    # Dunder methods
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ProblemInstance):
            return False

        return self.task_instance == value.task_instance

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.task_instance,
            self.n_tasks,
            self.n_jobs,
            self._n_machines,
            self.job_ids,
            self.job_tasks,
            self.preemptive,
            self.optional,
            self.processing_times,
        )
        return (self.__class__.__new__, (self.__class__,), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.task_instance,
            self.n_tasks,
            self.n_jobs,
            self._n_machines,
            self.job_ids,
            self.job_tasks,
            self.preemptive,
            self.optional,
            self.processing_times,
        ) = state


class TaskHistory:
    assignment: MACHINE_ID
    start_time: TIME
    duration: TIME
    end_time: TIME

    def __init__(
        self, assignment: MACHINE_ID, start_time: TIME, duration: TIME
    ) -> None:
        self.assignment = assignment
        self.start_time = start_time
        self.duration = duration
        self.end_time = start_time + duration

    def __repr__(self) -> str:
        return (
            f"TaskHistory(assignment={self.assignment}, "
            f"start_time={self.start_time}, "
            f"duration={self.duration}, "
            f"end_time={self.end_time})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            (self.assignment, self.start_time, self.duration),
            (),
        )


class Bounds:
    """
    Container for integer bounds in the scheduling environment.

    Each variable (e.g., start time, end time) in a Constraint Programming model
    has a domain set of values that are consistent with the constraints of the
    problem.

    The Bounds class maintains interval domains for each variable, managing both
    lower and upper bounds for each task-machine pair, as well as global bounds
    for each task, defined as

    - global_lb(task) = min(lb(task, machine) for machine in machines)
    - global_ub(task) = max(ub(task, machine) for machine in machines)

    ## IMPORTANT
    Never acess or modify the bounds directly outside of ScheduleState to ensure
    consistency.
    """

    __slots__ = (
        "n_machines",
        "lbs",
        "global_lbs",
        "ubs",
        "global_ubs",
    )

    n_machines: int

    lbs: list[TIME]
    global_lbs: list[TIME]

    ubs: list[TIME]
    global_ubs: list[TIME]

    def __init__(self, instance: ProblemInstance) -> None:
        n_tasks = instance.n_tasks
        n_machines = instance.n_machines

        self.n_machines = n_machines

        self.lbs = [MAX_TIME] * (n_tasks * n_machines)
        self.global_lbs = [MAX_TIME] * n_tasks

        self.ubs = [MIN_TIME] * (n_tasks * n_machines)
        self.global_ubs = [MIN_TIME] * n_tasks

        for task_id, p_times in enumerate(instance.processing_times):
            if not p_times:
                continue

            for machine_id in p_times:
                self.lbs[task_id * n_machines + machine_id] = MIN_TIME
                self.ubs[task_id * n_machines + machine_id] = MAX_TIME

            self.global_lbs[task_id] = MIN_TIME
            self.global_ubs[task_id] = MAX_TIME

    def recompute_global_bounds(self, task_id: TASK_ID) -> None:
        start = task_id * self.n_machines
        end = start + self.n_machines

        min_lb = self.lbs[start]
        max_ub = self.ubs[start]

        for i in range(start + 1, end):
            if self.lbs[i] < min_lb:
                min_lb = self.lbs[i]

            if self.ubs[i] > max_ub:
                max_ub = self.ubs[i]

        self.global_lbs[task_id] = min_lb
        self.global_ubs[task_id] = max_ub

    def __repr__(self) -> str:
        return (
            f"Bounds(lbs={self.lbs}, "
            f"ubs={self.ubs}, "
            f"global_lbs={self.global_lbs}, "
            f"global_ubs={self.global_ubs})"
        )

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

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.n_machines,
            self.lbs,
            self.global_lbs,
            self.ubs,
            self.global_ubs,
        )
        return (self.__class__, (), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.n_machines,
            self.lbs,
            self.global_lbs,
            self.ubs,
            self.global_ubs,
        ) = state


class ScheduleVariables:
    """
    Container for the task variables in the scheduling environment.

    Do not modify these variables directly, use the appropriate methods in
    ScheduleState to ensure consistency and proper updates of the bounds and
    feasibility checks.
    """

    __slots__ = [
        "remaining_times",
        "assignment",
        "presence",
        "start",
        "end"
    ]

    remaining_times: list[TIME]
    assignment: list[MACHINE_ID]

    presence: list[PresenceType]

    start: Bounds
    end: Bounds

    def __init__(self, instance: ProblemInstance) -> None:
        n_tasks = instance.n_tasks
        n_machines = instance.n_machines

        self.remaining_times = [MAX_TIME] * (n_tasks * n_machines)
        self.assignment = [GLOBAL_MACHINE_ID] * n_tasks

        self.presence = [
            Presence.UNDEFINED if optional else Presence.PRESENT
            for optional in instance.optional
        ]

        self.start = Bounds(instance)
        self.end = Bounds(instance)

        for task_id, p_times in enumerate(instance.processing_times):
            for machine, processing_time in p_times.items():
                idx = task_id * n_machines + machine

                self.remaining_times[idx] = processing_time
                self.start.ubs[idx] = self.end.ubs[idx] - processing_time
                self.end.lbs[idx] = self.start.lbs[idx] + processing_time

            self.start.recompute_global_bounds(task_id)
            self.end.recompute_global_bounds(task_id)

    def __repr__(self) -> str:
        return (
            f"ScheduleVariables(remaining_times={self.remaining_times}, "
            f"assignment={self.assignment}, presence={self.presence}, "
            f"start={self.start}, end={self.end})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.remaining_times,
            self.assignment,
            self.presence,
            self.start,
            self.end,
        )
        return (self.__class__, (), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.remaining_times,
            self.assignment,
            self.presence,
            self.start,
            self.end,
        ) = state

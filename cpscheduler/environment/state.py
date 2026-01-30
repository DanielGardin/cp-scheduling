from typing import Any
from collections import deque

from cpscheduler.environment._common import (
    MAX_TIME,
    MIN_TIME,
    MACHINE_ID,
    TASK_ID,
    TIME,
    STATUS,
    StatusEnum,
    ObsType,
    GLOBAL_MACHINE_ID
)
from cpscheduler.environment.events import VarField, Event
from cpscheduler.environment.tasks import Task, Job, TaskHistory
from cpscheduler.utils.list_utils import convert_to_list


def check_instance_consistency(instance: dict[str, list[Any]]) -> int:
    "Check if all lists in the instance have the same length."
    lengths = {len(v) for v in instance.values()}

    if len(lengths) > 1:
        raise ValueError("Inconsistent instance data: all lists must have the same length.")

    return lengths.pop() if lengths else 0


def min_bound(bound: dict[MACHINE_ID, TIME]) -> TIME:
    min_value = MAX_TIME

    for machine, value in bound.items():
        if machine != GLOBAL_MACHINE_ID and value < min_value:
            min_value = value

    return min_value


def max_bound(bound: dict[MACHINE_ID, TIME]) -> TIME:
    max_value = MIN_TIME

    for machine, value in bound.items():
        if machine != GLOBAL_MACHINE_ID and value > max_value:
            max_value = value

    return max_value


class ScheduleState:
    tasks_to_propagate: None

    tasks: list[Task]
    jobs: list[Job]

    time: TIME
    event_queue: deque[Event]

    awaiting_tasks: set[Task]
    fixed_tasks: set[Task]

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

        self.instance = {}
        self._n_machines = 0

    # def __repr__(self) -> str:
    #     cls_name = self.__class__.__name__

    #     if self.loaded:
    #         return (
    #             f"{cls_name}("
    #             f"n_machines={self.n_machines}, "
    #             f"n_jobs={self.n_jobs}, "
    #             f"n_tasks={self.n_tasks}, "
    #             f"awaiting={len(self.awaiting_tasks)}, "
    #             f"fixed={len(self.fixed_tasks)}"
    #             f")"
    #         )

    #     return f"{cls_name}(loaded={self.loaded})"

    # def __eq__(self, other: Any) -> bool:
    #     if not isinstance(other, ScheduleState):
    #         return False

    #     return (
    #         self.tasks == other.tasks
    #         and self.jobs == other.jobs
    #         and self.awaiting_tasks == other.awaiting_tasks
    #         and self.tasks_to_propagate == other.tasks_to_propagate
    #         and self.fixed_tasks == other.fixed_tasks
    #         and self.instance == other.instance
    #         and self.job_instance == other.job_instance
    #         and self.n_machines == other.n_machines
    #     )

    # def __reduce__(self) -> Any:
    #     return (
    #         self.__class__,
    #         (),
    #         (
    #             self.tasks,
    #             self.jobs,
    #             self.awaiting_tasks,
    #             self.tasks_to_propagate,
    #             self.fixed_tasks,
    #             self.instance,
    #             self.job_instance,
    #             self.n_machines,
    #         ),
    #     )

    # def __setstate__(self, state: tuple[Any, ...]) -> None:
    #     (
    #         self.tasks,
    #         self.jobs,
    #         self.awaiting_tasks,
    #         self.tasks_to_propagate,
    #         self.fixed_tasks,
    #         self.instance,
    #         self.job_instance,
    #         self.n_machines,
    #     ) = state

    @property
    def n_machines(self) -> int:
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

    def clear(self) -> None:
        self.tasks.clear()
        self.jobs.clear()

        self.time = 0
        self.event_queue.clear()

        self.awaiting_tasks.clear()
        self.fixed_tasks.clear()

        self.instance.clear()
        self._n_machines = 0

    def reset(self) -> None:
        self.time = 0

        for task in self.tasks: task.reset()

        self.awaiting_tasks.update(self.tasks)
        self.event_queue.clear()
        self.fixed_tasks.clear()

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
            self.awaiting_tasks.add(task)

        self.instance["task_id"] = list(range(n_tasks))
        self.instance["job_id"] = job_ids
        self.job_instance["job_id"] = list(range(self.n_jobs))

    def tight_start_lb(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        task = self.tasks[task_id]

        changed: bool = False
        if machine_id != GLOBAL_MACHINE_ID:
            if task.get_start_lb(machine_id) < value:
                task.start_lbs_[machine_id] = value
                self.event_queue.append(Event(task, VarField.START_LB, machine_id))
                changed = True

        else:
            for machine in task.machines:
                if task.get_start_lb(machine) < value:
                    task.start_lbs_[machine] = value
                    changed = True

            if changed:
                self.event_queue.append(Event(task, VarField.START_LB))

        if changed:
            task.start_lbs_[GLOBAL_MACHINE_ID] = min_bound(task.start_lbs_)

    def tight_start_ub(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        task = self.tasks[task_id]

        changed: bool = False
        if machine_id != GLOBAL_MACHINE_ID:
            if task.get_start_ub(machine_id) > value:
                task.start_ubs_[machine_id] = value
                self.event_queue.append(Event(task, VarField.START_UB, machine_id))
                changed = True

        else:
            for machine in task.machines:
                if task.get_start_ub(machine) > value:
                    task.start_ubs_[machine] = value
                    changed = True
            
            if changed:
                self.event_queue.append(Event(task, VarField.START_UB))

        if changed:
            task.start_ubs_[GLOBAL_MACHINE_ID] = max_bound(task.start_ubs_)

    def tight_end_lb(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        task = self.tasks[task_id]

        changed: bool = False
        if machine_id != GLOBAL_MACHINE_ID:
            if task.get_end_lb(machine_id) < value:
                task.start_lbs_[machine_id] = value - task.remaining_times_[machine_id]
                self.event_queue.append(Event(task, VarField.END_LB, machine_id))
                changed = True

        else:
            for machine in task.machines:
                if task.get_end_lb(machine) < value:
                    task.start_lbs_[machine] = value - task.remaining_times_[machine]
                    changed = True
                
                if changed:
                    self.event_queue.append(Event(task, VarField.END_LB))
                    

        if changed:
            task.start_lbs_[GLOBAL_MACHINE_ID] = min_bound(task.start_lbs_)

    def tight_end_ub(
        self,
        task_id: TASK_ID,
        value: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> None:
        task = self.tasks[task_id]

        changed: bool = False
        if machine_id != GLOBAL_MACHINE_ID:
            if task.get_end_ub(machine_id) > value:
                task.start_ubs_[machine_id] = value - task.remaining_times_[machine_id]
                self.event_queue.append(Event(task, VarField.END_UB, machine_id))
                changed = True

        else:
            for machine in task.machines:
                if task.get_end_ub(machine) > value:
                    task.start_ubs_[machine] = value - task.remaining_times_[machine]
                    changed = True
            
            if changed:
                self.event_queue.append(Event(task, VarField.END_UB))

        if changed:
            task.start_ubs_[GLOBAL_MACHINE_ID] = max_bound(task.start_ubs_)

    def set_infeasible(self, task_id: TASK_ID) -> None:
        self.tight_start_lb(task_id, MAX_TIME)
        self.tight_start_ub(task_id, MIN_TIME)

        if not self.tasks[task_id].optional:
            self.infeasible = True

    def execute_task(
        self,
        task_id: TASK_ID,
        current_time: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> bool:
        task = self.tasks[task_id]

        if not task.is_available(current_time, machine_id):
            return False

        if machine_id == GLOBAL_MACHINE_ID:
            for machine in task.machines:
                if task.is_available(current_time, machine):
                    machine_id = machine
                    break

            else:
                return False

        self.tight_start_lb(task_id, current_time, machine_id)
        self.tight_start_ub(task_id, current_time, machine_id)

        task.assignment_ = machine_id
        task.fixed_ = True

        self.awaiting_tasks.remove(task)
        self.fixed_tasks.add(task)

        history_entry = TaskHistory(
            assignment=machine_id,
            start_time=current_time,
            duration=task.remaining_times_[machine_id],
            end_time=current_time + task.remaining_times_[machine_id],
        )
        task.history.append(history_entry)

        return True

    def pause_task(
        self,
        task_id: TASK_ID,
        current_time: TIME,
    ) -> bool:
        task = self.tasks[task_id]

        if not task.fixed_ or not task.preemptive:
            return False

        cur_processing_time = task.processing_times[task.assignment_]
        actual_time = current_time - task.get_start_lb(task.assignment_)

        if actual_time >= cur_processing_time:
            return False

        for machine, prev_time in task.remaining_times_.items():
            work_done = (prev_time * actual_time) // cur_processing_time
            task.remaining_times_[machine] -= work_done

            task.start_lbs_[machine] = current_time
            task.start_ubs_[machine] = MAX_TIME

        task.start_lbs_[GLOBAL_MACHINE_ID] = current_time
        task.start_ubs_[GLOBAL_MACHINE_ID] = MAX_TIME

        task.fixed_ = False
        task.assignment_ = GLOBAL_MACHINE_ID

        history_entry = task.history.pop()

        task.history.append(
            TaskHistory(
                assignment=history_entry.assignment,
                start_time=history_entry.start_time,
                duration=actual_time,
                end_time=history_entry.start_time + actual_time,
            )
        )

        self.awaiting_tasks.add(task)
        self.fixed_tasks.remove(task)

        return True

    def execute_job(
        self,
        job_id: TASK_ID,
        current_time: TIME,
        machine_id: MACHINE_ID = GLOBAL_MACHINE_ID,
    ) -> TASK_ID:
        for task in self.jobs[job_id]:
            if self.execute_task(task.task_id, current_time, machine_id):
                return task.task_id

        return -1

    def pause_job(
        self,
        job_id: TASK_ID,
        current_time: TIME,
    ) -> TASK_ID:
        for task in self.jobs[job_id]:
            if self.pause_task(task.task_id, current_time):
                return task.task_id

        return -1

    def get_status(self, time: TIME) -> list[STATUS]:
        status_list: list[STATUS] = []

        status: STATUS = StatusEnum.UNFEASIBLE
        for task in self.tasks:
            if not task.fixed_:
                if time >= task.get_start_ub(GLOBAL_MACHINE_ID) or not task.is_feasible(time):
                    status = StatusEnum.UNFEASIBLE

                elif task.history:
                    status = StatusEnum.PAUSED

                else:
                    status = StatusEnum.AWAITING

            else:
                history_entry = task.history[-1]
                start_time = history_entry.start_time
                end_time = history_entry.end_time

                if time < start_time:
                    status = StatusEnum.AWAITING

                elif start_time <= time < end_time:
                    status = StatusEnum.EXECUTING

                elif time >= end_time:
                    status = StatusEnum.COMPLETED

            status_list.append(status)

        return status_list

    def get_observation(self, time: TIME) -> ObsType:
        task_obs = self.instance.copy()
        job_obs = self.job_instance.copy()

        task_obs["status"] = self.get_status(time)
        task_obs["available"] = [task.is_available(time) for task in self.tasks]

        return task_obs, job_obs

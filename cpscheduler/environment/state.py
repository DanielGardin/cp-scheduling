from typing import Any

from cpscheduler.environment._common import MACHINE_ID, TASK_ID, TIME, ObsType
from cpscheduler.environment.tasks import Task
from cpscheduler.utils.list_utils import convert_to_list


def check_instance_consistency(instance: dict[str, list[Any]]) -> int:
    "Check if all lists in the instance have the same length."
    lengths = {len(v) for v in instance.values()}

    if len(lengths) > 1:
        raise ValueError(
            "Inconsistent instance data: all lists must have the same length."
        )

    return lengths.pop() if lengths else 0


# TODO: Manage task and job data
class ScheduleState:
    tasks: list[Task]
    jobs: list[list[Task]]

    awaiting_tasks: set[Task]
    transition_tasks: set[Task]
    fixed_tasks: set[Task]

    instance: dict[str, list[Any]]
    job_instance: dict[str, list[Any]]
    n_machines: int

    def __init__(self) -> None:
        self.tasks = []
        self.jobs = []

        self.awaiting_tasks = set()
        self.transition_tasks = set()
        self.fixed_tasks = set()

        self.instance = {}
        self.n_machines = 0

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__

        if self.loaded:
            return (
                f"{cls_name}("
                f"n_machines={self.n_machines}, "
                f"n_jobs={self.n_jobs}, "
                f"n_tasks={self.n_tasks}, "
                f"awaiting={len(self.awaiting_tasks)}, "
                f"transition={len(self.transition_tasks)}, "
                f"fixed={len(self.fixed_tasks)}"
                f")"
            )

        return f"{cls_name}(loaded={self.loaded})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ScheduleState):
            return False

        return (
            self.tasks == other.tasks
            and self.jobs == other.jobs
            and self.awaiting_tasks == other.awaiting_tasks
            and self.transition_tasks == other.transition_tasks
            and self.fixed_tasks == other.fixed_tasks
            and self.instance == other.instance
            and self.job_instance == other.job_instance
            and self.n_machines == other.n_machines
        )

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (),
            (
                self.tasks,
                self.jobs,
                self.awaiting_tasks,
                self.transition_tasks,
                self.fixed_tasks,
                self.instance,
                self.job_instance,
                self.n_machines,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.tasks,
            self.jobs,
            self.awaiting_tasks,
            self.transition_tasks,
            self.fixed_tasks,
            self.instance,
            self.job_instance,
            self.n_machines,
        ) = state

    def set_n_machines(self, n: int) -> None:
        self.n_machines = n

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

        self.awaiting_tasks.clear()
        self.transition_tasks.clear()
        self.fixed_tasks.clear()

        self.instance.clear()

    def reset(self) -> None:
        for task in self.tasks:
            task.reset()
            self.awaiting_tasks.add(task)

        self.transition_tasks.clear()
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

        for _ in set(job_ids):
            self.jobs.append([])

        for task_id, job_id in enumerate(job_ids):
            task = Task(task_id, job_id)

            self.tasks.append(task)
            self.jobs[job_id].append(task)
            self.awaiting_tasks.add(task)

        self.instance["task_id"] = list(range(n_tasks))
        self.instance["job_id"] = job_ids
        self.job_instance["job_id"] = list(range(self.n_jobs))

    def get_job_completion_time(self, job_id: TASK_ID, time: TIME) -> TIME:
        return max(
            task.get_end() for task in self.jobs[job_id] if task.is_completed(time)
        )

    def execute_task(
        self,
        task_id: TASK_ID,
        current_time: TIME,
        machine_id: MACHINE_ID = -1,
    ) -> bool:
        task = self.tasks[task_id]

        executed = task.execute(current_time, machine_id)

        if executed:
            self.awaiting_tasks.remove(task)
            self.transition_tasks.add(task)
            self.fixed_tasks.add(task)

        return executed

    def pause_task(
        self,
        task_id: TASK_ID,
        current_time: TIME,
    ) -> bool:
        task = self.tasks[task_id]

        paused = task.pause(current_time)

        if paused:
            self.awaiting_tasks.add(task)
            self.transition_tasks.add(task)
            self.fixed_tasks.remove(task)

        return paused

    def execute_job(
        self,
        job_id: TASK_ID,
        current_time: TIME,
        machine_id: MACHINE_ID = -1,
    ) -> TASK_ID:
        for task in self.jobs[job_id]:
            if task.execute(current_time, machine_id):
                self.awaiting_tasks.remove(task)
                self.transition_tasks.add(task)
                self.fixed_tasks.add(task)
                return task.task_id

        return -1

    def pause_job(
        self,
        job_id: TASK_ID,
        current_time: TIME,
    ) -> TASK_ID:
        for task in self.jobs[job_id]:
            if task.pause(current_time):
                self.awaiting_tasks.add(task)
                self.transition_tasks.add(task)
                self.fixed_tasks.remove(task)
                return task.task_id

        return -1

    def get_observation(self, time: TIME) -> ObsType:
        task_obs = self.instance.copy()
        job_obs = self.job_instance.copy()

        task_obs["status"] = []
        task_obs["available"] = []

        for task in self.tasks:
            task_obs["status"].append(task.get_status(time))
            task_obs["available"].append(task.is_available(time))

        return task_obs, job_obs

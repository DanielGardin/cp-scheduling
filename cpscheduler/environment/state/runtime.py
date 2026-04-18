from cpscheduler.environment.constants import (
    MachineID, TaskID, Time, Status, StatusType,
    MIN_TIME,
    EzPickle, CustomDataclass
)

from cpscheduler.environment.state.instance import ProblemInstance

DUMMY_INSTANCE = ProblemInstance({})

AWAITING = Status.AWAITING

class TaskHistory(CustomDataclass):
    "A record of a task execution, (machine_id, start_time, end_time)"

    machine_id: MachineID
    start_time: Time
    end_time: Time

    def __init__(self, machine_id: MachineID, start_time: Time, end_time: Time):
        self.machine_id = machine_id
        self.start_time = start_time
        self.end_time = end_time


class RuntimeState(EzPickle):
    """
    Container for the runtime state of the scheduling environment.

    This class is used to store any additional state information that may be
    needed by constraints or other components of the environment. It is designed
    to be flexible and can be extended with additional attributes as needed.
    """

    __slots__ = (
        "history",
        "prerequisites",
        "status",
        "awaiting_tasks",
        "unlocked_tasks",
        "executing_tasks",
        "completed_tasks",
        "last_completion_time"
    )

    history: list[list[TaskHistory]]

    prerequisites: list[set[str]]
    status: list[StatusType]

    awaiting_tasks: set[TaskID]
    unlocked_tasks: set[TaskID]
    executing_tasks: set[TaskID]
    completed_tasks: set[TaskID]

    last_completion_time: Time

    def __init__(self, instance: ProblemInstance | None = None) -> None:
        if instance is None:
            instance = DUMMY_INSTANCE

        n_tasks = instance.n_tasks

        self.history = [[] for _ in range(n_tasks)]

        self.prerequisites = [set() for _ in range(n_tasks)]
        self.status = [AWAITING] * n_tasks

        self.awaiting_tasks = set(range(n_tasks))
        self.unlocked_tasks = set(range(n_tasks))
        self.executing_tasks = set()
        self.completed_tasks = set()

        self.last_completion_time = MIN_TIME

    def __repr__(self) -> str:
        return (
            f"RuntimeState(history={self.history}, "
            f"executing_tasks={self.executing_tasks}, "
            f"completed_tasks={self.completed_tasks})"
        )

    def get_assignment(self, task_id: TaskID, page: int = -1) -> MachineID:
        "Get the machine assignment of the current execution of a given task."
        return self.history[task_id][page].machine_id

    def get_start(self, task_id: TaskID, page: int = -1) -> Time:
        "Get the start time of the current execution of a given task."
        return self.history[task_id][page].start_time

    def get_end(self, task_id: TaskID, page: int = -1) -> Time:
        "Get the end time of the current execution of a given task."
        return self.history[task_id][page].end_time

    def get_history(self, task_id: TaskID, page: int = -1) -> TaskHistory:
        "Get the execution history for a given task and page."
        return self.history[task_id][page]

    def get_awaiting_tasks(self) -> list[TaskID]:
        return list(self.awaiting_tasks)

    def get_unlocked_tasks(self) -> list[TaskID]:
        return list(self.unlocked_tasks)

    def get_executing_tasks(self) -> list[TaskID]:
        return list(self.executing_tasks)

    def get_completed_tasks(self) -> list[TaskID]:
        return list(self.completed_tasks)

    def recompute_last_completion_time(self) -> None:
        best = MIN_TIME

        for task_id in self.executing_tasks:
            completion_time = self.get_end(task_id)

            if completion_time > best:
                best = completion_time
        
        self.last_completion_time = best

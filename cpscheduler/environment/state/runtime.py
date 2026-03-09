from typing import Any, TypeAlias

from cpscheduler.environment.constants import (
    MachineID,
    TaskID,
    Time,
    Status,
    StatusType,
)

from cpscheduler.environment.state.des import ProblemInstance

DUMMY_INSTANCE = ProblemInstance({})

TaskHistory: TypeAlias = tuple[MachineID, Time, Time]
"A record of a task execution, (machine_id, start_time, end_time)"

class RuntimeState:
    """
    Container for the runtime state of the scheduling environment.

    This class is used to store any additional state information that may be
    needed by constraints or other components of the environment. It is designed
    to be flexible and can be extended with additional attributes as needed.
    """

    __slots__ = (
        "history",
        "awaiting_tasks",
        "executing_tasks",
        "completed_tasks",
        "status",
        "last_completion_time",
    )

    history: list[list[TaskHistory]]

    awaiting_tasks: set[TaskID]
    executing_tasks: set[TaskID]
    completed_tasks: set[TaskID]

    status: list[StatusType]

    last_completion_time: Time

    def __init__(self, instance: ProblemInstance) -> None:
        self.history = [[] for _ in range(instance.n_tasks)]

        self.awaiting_tasks = set(range(instance.n_tasks))
        self.executing_tasks = set()
        self.completed_tasks = set()

        self.status = [Status.AWAITING] * instance.n_tasks
        self.last_completion_time = 0

    def __repr__(self) -> str:
        return (
            f"RuntimeState(history={self.history}, "
            f"awaiting_tasks={self.awaiting_tasks}, "
            f"executing_tasks={self.executing_tasks}, "
            f"completed_tasks={self.completed_tasks})"
        )

    def get_assignment(self, task_id: TaskID, page: int = -1) -> MachineID:
        "Get the machine assignment of the current execution of a given task."
        return self.history[task_id][page][0]

    def get_start(self, task_id: TaskID, page: int = -1) -> Time:
        "Get the start time of the current execution of a given task."
        return self.history[task_id][page][1]

    def get_end(self, task_id: TaskID, page: int = -1) -> Time:
        "Get the end time of the current execution of a given task."
        return self.history[task_id][page][2]

    def get_history(self, task_id: TaskID, page: int = -1) -> TaskHistory:
        "Get the execution history for a given task and page."
        return self.history[task_id][page]

    def is_terminal(self) -> bool:
        "Check if no tasks are currently executing or awaiting"
        return not self.awaiting_tasks and not self.executing_tasks

    def start_task(
        self,
        task_id: TaskID,
        machine_id: MachineID,
        start_time: Time,
        end_time: Time,
    ) -> None:
        self.awaiting_tasks.discard(task_id)
        self.executing_tasks.add(task_id)

        self.history[task_id].append((machine_id, start_time, end_time))

        self.status[task_id] = Status.EXECUTING

        if end_time > self.last_completion_time:
            self.last_completion_time = end_time

    def pause_task(self, task_id: TaskID, time: Time) -> None:
        self.executing_tasks.discard(task_id)
        self.awaiting_tasks.add(task_id)

        assignment, start_time, prev_end = self.history[task_id].pop()

        self.history[task_id].append((assignment, start_time, time))

        self.status[task_id] = Status.PAUSED

        if prev_end == self.last_completion_time:
            self.last_completion_time = max(
                (self.get_end(t_id) for t_id in self.executing_tasks), default=0
            )

    def infeasible_task(self, task_id: TaskID) -> None:
        self.awaiting_tasks.discard(task_id)

        self.status[task_id] = Status.INFEASIBLE

    def update(self, time: Time) -> None:
        history = self.history

        for task_id in list(self.executing_tasks):
            if history[task_id][-1][2] <= time:
                self.executing_tasks.remove(task_id)
                self.completed_tasks.add(task_id)

                self.status[task_id] = Status.COMPLETED

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, RuntimeState):
            return NotImplemented

        return (
            self.history == value.history
            and self.awaiting_tasks == value.awaiting_tasks
            and self.executing_tasks == value.executing_tasks
            and self.completed_tasks == value.completed_tasks
            and self.status == value.status
        )

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.history,
            self.awaiting_tasks,
            self.executing_tasks,
            self.completed_tasks,
            self.status,
            self.last_completion_time,
        )
        return (self.__class__, (DUMMY_INSTANCE,), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.history,
            self.awaiting_tasks,
            self.executing_tasks,
            self.completed_tasks,
            self.status,
            self.last_completion_time,
        ) = state

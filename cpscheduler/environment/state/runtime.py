from typing import Any, Final, Literal, assert_never
from typing_extensions import NamedTuple

from cpscheduler.environment.constants import (
    MachineID,
    TaskID,
    Time,
    Status,
    StatusType,
    MIN_TIME,
    GLOBAL_MACHINE_ID
)

from cpscheduler.environment.state.instance import ProblemInstance

DUMMY_INSTANCE = ProblemInstance({})

AWAITING = Status.AWAITING
PAUSED = Status.PAUSED
EXECUTING = Status.EXECUTING
COMPLETED = Status.COMPLETED


EventKindType = Literal[0, 1, 2]

class RuntimeEventKind:
    __slots__ = ()

    TASK_STARTED: Final[Literal[0]] = 0
    TASK_PAUSED: Final[Literal[1]] = 1
    TASK_COMPLETED: Final[Literal[2]] = 2

TASK_STARTED = RuntimeEventKind.TASK_STARTED
TASK_PAUSED = RuntimeEventKind.TASK_PAUSED
TASK_COMPLETED = RuntimeEventKind.TASK_COMPLETED


def kind_to_str(kind: EventKindType) -> str:
    if kind == TASK_STARTED:
        return "TASK_STARTED"

    if kind == TASK_PAUSED:
        return "TASK_PAUSED"
    
    if kind == TASK_COMPLETED:
        return "TASK_COMPLETED"
    
    assert_never(kind)


class RuntimeEvent:
    """
    Container for runtime events in the scheduling environment.
    """

    __slots__ = ("task_id", "kind", "machine_id")

    task_id: TaskID
    kind: EventKindType
    machine_id: MachineID

    def __init__(
        self,
        task_id: TaskID,
        kind: EventKindType,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        self.task_id = task_id
        self.kind = kind
        self.machine_id = machine_id

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            (self.task_id, self.kind, self.machine_id),
        )

    def __repr__(self) -> str:
        string = f"DomainEvent(task_id={self.task_id}, kind={kind_to_str(self.kind)}"

        if self.machine_id != GLOBAL_MACHINE_ID:
            string += f", machine_id={self.machine_id}"

        return string + ")"



class TaskHistory(NamedTuple):
    "A record of a task execution, (machine_id, start_time, end_time)"

    machine_id: MachineID
    start_time: Time
    end_time: Time


class RuntimeState:
    """
    Container for the runtime state of the scheduling environment.

    This class is used to store any additional state information that may be
    needed by constraints or other components of the environment. It is designed
    to be flexible and can be extended with additional attributes as needed.
    """

    __slots__ = (
        "history",
        "executing_tasks",
        "completed_tasks",
        "status",
        "last_completion_time",
        "event_queue"
    )

    history: list[list[TaskHistory]]

    executing_tasks: set[TaskID]
    completed_tasks: set[TaskID]

    status: list[StatusType]

    last_completion_time: Time

    event_queue: list[RuntimeEvent]

    def __init__(self, instance: ProblemInstance) -> None:
        self.history = [[] for _ in range(instance.n_tasks)]

        self.executing_tasks = set()
        self.completed_tasks = set()

        self.status = [AWAITING] * instance.n_tasks
        self.last_completion_time = MIN_TIME

        self.event_queue = []

    def __repr__(self) -> str:
        return (
            f"RuntimeState(history={self.history}, "
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

    def start_task(
        self,
        task_id: TaskID,
        machine_id: MachineID,
        start_time: Time,
        end_time: Time,
    ) -> None:
        self.executing_tasks.add(task_id)

        self.history[task_id].append(
            TaskHistory(machine_id, start_time, end_time)
        )

        self.status[task_id] = EXECUTING
        self.event_queue.append(
            RuntimeEvent(task_id, TASK_STARTED, machine_id)
        )

        if end_time > self.last_completion_time:
            self.last_completion_time = end_time

    def pause_task(self, task_id: TaskID, time: Time) -> None:
        self.executing_tasks.discard(task_id)

        assignment, start_time, prev_end = self.history[task_id].pop()

        self.history[task_id].append(TaskHistory(assignment, start_time, time))

        self.status[task_id] = PAUSED
        self.event_queue.append(
            RuntimeEvent(task_id, TASK_PAUSED, assignment)
        )

        if prev_end == self.last_completion_time:
            self.last_completion_time = max(
                (self.get_end(t_id) for t_id in self.executing_tasks), default=0
            )

    def update(self, time: Time) -> None:
        history = self.history

        for task_id in list(self.executing_tasks):
            assignment, _, end_time = history[task_id][-1]

            if end_time <= time:
                self.executing_tasks.remove(task_id)
                self.completed_tasks.add(task_id)

                self.status[task_id] = COMPLETED
                self.event_queue.append(
                    RuntimeEvent(task_id, TASK_COMPLETED, assignment)
                )


    def __eq__(self, value: object) -> bool:
        if not isinstance(value, RuntimeState):
            return NotImplemented

        return (
            self.history == value.history
            and self.executing_tasks == value.executing_tasks
            and self.completed_tasks == value.completed_tasks
            and self.status == value.status
        )

    def __reduce__(self) -> tuple[Any, ...]:
        state = (
            self.history,
            self.executing_tasks,
            self.completed_tasks,
            self.status,
            self.last_completion_time,
            self.event_queue,
        )
        return (self.__class__, (DUMMY_INSTANCE,), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            self.history,
            self.executing_tasks,
            self.completed_tasks,
            self.status,
            self.last_completion_time,
            self.event_queue,
        ) = state

"""Runtime containers for the scheduling DES."""

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import (
    MIN_TIME,
    EzPickle,
    MachineID,
    Status,
    StatusType,
    TaskID,
    Time,
)
from cpscheduler.environment.instance import ProblemInstance

AWAITING = Status.AWAITING


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class TaskHistory(EzPickle):
    """Record of a task execution segment.

    A task can be executed multiple times in preemptive problems.
    Each history entry records the machine, start and end times for one execution segment.

    """

    machine_id: MachineID
    start_time: Time
    end_time: Time

    def __init__(self, machine_id: MachineID, start_time: Time, end_time: Time):
        """Initialize a TaskHistory entry.

        Parameters
        ----------
        machine_id : MachineID
            Machine where the segment executed.

        start_time : Time
            Inclusive start time of the segment.

        end_time : Time
            Exclusive end time of the segment.

        """
        self.machine_id = machine_id
        self.start_time = start_time
        self.end_time = end_time

    def __eq__(self, value: object, /) -> bool:
        """Check equality of TaskHistory containers."""
        return (
            isinstance(value, TaskHistory)
            and self.machine_id == value.machine_id
            and self.start_time == value.start_time
            and self.end_time == value.end_time
        )


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class RuntimeState(EzPickle):
    """Aggregate container for runtime variables used during simulation."""

    history: list[list[TaskHistory]]

    dependencies: list[set[str]]
    status: list[StatusType]

    awaiting_tasks: set[TaskID]
    unlocked_tasks: set[TaskID]
    executing_tasks: set[TaskID]
    completed_tasks: set[TaskID]

    last_completion_time: Time

    def __init__(self, instance: ProblemInstance) -> None:
        """Initialize the RuntimeState with a problem instance.

        Parameters
        ----------
        instance: ProblemInstance
            The problem instance containing tasks, machines, processing times, etc.

        """
        n_tasks = instance.n_tasks

        self.history = [[] for _ in range(n_tasks)]

        self.dependencies = [set() for _ in range(n_tasks)]
        self.status = [AWAITING] * n_tasks

        self.awaiting_tasks = set(range(n_tasks))
        self.unlocked_tasks = set(range(n_tasks))
        self.executing_tasks = set()
        self.completed_tasks = set()

        self.last_completion_time = MIN_TIME

    def get_assignment(self, task_id: TaskID, page: int = -1) -> MachineID:
        """Return the machine assignment for the current execution page of a task."""
        return self.history[task_id][page].machine_id

    def get_start(self, task_id: TaskID, page: int = -1) -> Time:
        """Return the start time for the given history page."""
        return self.history[task_id][page].start_time

    def get_end(self, task_id: TaskID, page: int = -1) -> Time:
        """Return the end time for the given history page."""
        return self.history[task_id][page].end_time

    def get_history(self, task_id: TaskID, page: int = -1) -> TaskHistory:
        """Return the TaskHistory for the given page."""
        return self.history[task_id][page]

    def recompute_last_completion_time(self) -> None:
        """Recompute `last_completion_time` from currently executing tasks."""
        best = MIN_TIME

        for task_id in self.executing_tasks:
            completion_time = self.get_end(task_id)

            if completion_time > best:
                best = completion_time

        self.last_completion_time = best

    def __eq__(self, value: object, /) -> bool:
        """Check equality of RuntimeState containers."""
        return (
            isinstance(value, RuntimeState)
            and self.history == value.history
            and self.dependencies == value.dependencies
            and self.status == value.status
            and self.awaiting_tasks == value.awaiting_tasks
            and self.unlocked_tasks == value.unlocked_tasks
            and self.executing_tasks == value.executing_tasks
            and self.completed_tasks == value.completed_tasks
            and self.last_completion_time == value.last_completion_time
        )

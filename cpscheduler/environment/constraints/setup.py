from collections.abc import Mapping

from cpscheduler.environment.constants import Time, TaskID, MachineID, Int
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint

class SetupConstraint(Constraint):
    """
    Setup constraint for the scheduling environment.

    This constraint is used to define the setup time between tasks.
    The setup times can be defined as a mapping of task IDs to a mapping of child task IDs
    and their respective setup times, or as a string that refers to a column in the tasks data.

    Arguments:
        setup_times: Mapping[int, Mapping[int, int]] | Callable[[int, int, ScheduleState], int]
            A mapping of task IDs to a mapping of child task IDs and their respective setup times.
            Alternatively, a callable function that takes in two task IDs and the scheduling data,
            and returns the setup time between the two tasks.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    setup_times: dict[TaskID, dict[TaskID, Time]]
    current_setup_times: dict[TaskID, dict[TaskID, Time]]

    def __init__(
            self,
            setup_times: Mapping[Int, Mapping[Int, Int]] | None = None
        ) -> None:
        if setup_times is None:
            setup_times = {}

        self.setup_times = {
            TaskID(task): {
                TaskID(child): Time(time)
                for child, time in children.items()
            }
            for task, children in setup_times.items()
        }

    def add_setup_time(
        self, task_id: Int, child_id: Int, setup_time: Int
    ) -> None:
        task = TaskID(task_id)
        child = TaskID(child_id)

        if task_id not in self.setup_times:
            self.setup_times[task] = {}

        self.setup_times[task][child] = Time(setup_time)

    def remove_setup_time(self, task_id: Int, child_id: Int) -> None:
        task = TaskID(task_id)
        child = TaskID(child_id)

        if task in self.setup_times and child in self.setup_times[task]:
            del self.setup_times[task][child]

            if not self.setup_times[task]:
                del self.setup_times[task]

    def reset(self, state: ScheduleState) -> None:
        self.current_setup_times = {
            task_id: children.copy()
            for task_id, children in self.setup_times.items()
        }

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        setup_times = self.current_setup_times

        for other_tasks in setup_times.values():
            other_tasks.pop(task_id, None)

        if task_id in setup_times:
            end_time = state.get_end_lb(task_id)
            setup_times_for_task = setup_times.pop(task_id)

            for child_id, setup_time in setup_times_for_task.items():
                state.tight_start_lb(child_id, end_time + setup_time)

from typing import Any, TypeAlias
from collections.abc import Mapping, Callable

from cpscheduler.environment.constants import TaskID, Time, Int
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint

SetupTimes: TypeAlias = (
    Mapping[Int, Mapping[Int, Int]] | Callable[[int, int, ScheduleState], Int]
)


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

    original_setup_times: dict[TaskID, dict[TaskID, Time]]
    setup_fn: Callable[[int, int, ScheduleState], Int] | None = None

    current_setup_times: dict[TaskID, dict[TaskID, Time]]

    def __init__(self, setup_times: SetupTimes) -> None:
        if callable(setup_times):
            self.setup_fn = setup_times
            self.original_setup_times = {}

        else:
            self.original_setup_times = {
                TaskID(task): {
                    TaskID(child): Time(time)
                    for child, time in children.items()
                }
                for task, children in setup_times.items()
            }

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.setup_fn or self.original_setup_times,),
            (self.current_setup_times,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.current_setup_times,) = state

    def initialize(self, state: ScheduleState) -> None:
        if self.setup_fn is None:
            return

        setup_times = {}
        n_tasks = state.n_tasks

        for task_id in range(n_tasks):
            task_setup_times = {}

            for child_id in range(n_tasks):
                if task_id == child_id:
                    continue

                setup_time = Time(self.setup_fn(task_id, child_id, state))

                if setup_time > 0:
                    task_setup_times[TaskID(child_id)] = setup_time

            setup_times[TaskID(task_id)] = task_setup_times

        self.original_setup_times = setup_times

    def reset(self, state: ScheduleState) -> None:
        self.current_setup_times = {
            task_id: children.copy()
            for task_id, children in self.original_setup_times.items()
        }

    def on_assignment(
        self, task_id: TaskID, machine_id: TaskID, state: ScheduleState
    ) -> None:
        setup_times = self.current_setup_times

        for other_tasks in setup_times.values():
            other_tasks.pop(task_id, None)

        if task_id in setup_times:
            end_time = state.get_end_lb(task_id)
            setup_times_for_task = setup_times.pop(task_id)

            for child_id, setup_time in setup_times_for_task.items():
                state.tight_start_lb(child_id, end_time + setup_time)

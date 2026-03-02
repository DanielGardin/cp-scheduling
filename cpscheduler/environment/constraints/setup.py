from typing import Any, TypeAlias
from collections.abc import Mapping, Callable

from cpscheduler.environment.constants import TaskID, Time, Int
from cpscheduler.environment.state.events import Event
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import Constraint

SetupTimes: TypeAlias = (
    Mapping[Int, Mapping[Int, Int]] | Callable[[int, int, Any], Int]
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
            (self.original_setup_times,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.original_setup_times,) = state

    def initialize(self, state: ScheduleState) -> None:
        if self.setup_fn is not None:
            self.original_setup_times.clear()

            for task_id in range(state.n_tasks):
                self.original_setup_times[TaskID(task_id)] = {}

                for child_id in range(state.n_tasks):
                    if task_id == child_id:
                        continue

                    setup_time = Time(self.setup_fn(task_id, child_id, state))

                    if setup_time > 0:
                        self.original_setup_times[TaskID(task_id)][
                            TaskID(child_id)
                        ] = setup_time

    def reset(self, state: ScheduleState) -> None:
        self.setup_times = {
            task_id: children.copy()
            for task_id, children in self.original_setup_times.items()
        }

    def propagate(self, event: Event, state: ScheduleState) -> None:
        task_id = event.task_id

        if not event.is_assignment() or task_id not in self.setup_times:
            return

        end_time = state.get_end_lb(task_id)

        for child_id in list(self.setup_times[task_id].keys()):
            # TODO: implement is_fixed
            if state.is_completed(child_id):
                self.setup_times[task_id].pop(child_id)
                continue

            setup_time = self.setup_times[task_id][child_id]

            state.tight_start_lb(child_id, end_time + setup_time)

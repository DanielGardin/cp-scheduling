from typing import Any
from collections.abc import Iterable

from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import RegularObjective

def makespan_(state: ScheduleState, tasks: Iterable[TaskID]) -> float:
    "Compute the makespan of a set of tasks."
    max_end_time: Time = 0

    for task in tasks:
        if not state.is_completed(task):
            continue

        end_time = state.get_end_ub(task)

        if end_time > max_end_time:
            max_end_time = end_time

    return float(max_end_time)


class Makespan(RegularObjective):
    """
    Classic makespan objective function, which aims to minimize the time at
    which all tasks are completed.
    """

    _value: Time

    def reset(self, state: ScheduleState) -> None:
        self._value = 0

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._value = state.time

    def get_current(self, state: ScheduleState) -> float:
        return float(self._value)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.minimize,),
            (self._value,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._value,) = state

    def __call__(self, state: ScheduleState) -> float:
        return makespan_(state, range(state.n_tasks))

    def get_entry(self) -> str:
        return "C_max"

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import MachineID, TaskID, EzPickle
from cpscheduler.environment.state import ScheduleState

objectives: dict[str, type["Objective"]] = {}

@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Objective(EzPickle):
    """
    Base class for all objective functions in the scheduling environment.

    Objective functions are used to evaluate the performance of a scheduling
    algorithm.
    They can be used to guide the search for an optimal schedule by providing a
    numerical value that represents the quality of the schedule.
    """

    minimize: bool

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        name = cls.__name__

        # Private classes are not registered
        if not name.startswith('_'):
            objectives[name] = cls

    def __init__(self, minimize: bool = True) -> None:
        self.minimize = minimize

    @property
    def regular(self) -> bool:
        "The objective is regular, when it is non-decreasing w.r.t completion times."
        return False

    def __repr__(self) -> str:
        sense = "minimize" if self.minimize else "maximize"

        return f"{type(self).__name__}(sense={sense})"

    def initialize(self, state: ScheduleState) -> None:
        "Initialize the objective with the given schedule state."

    def reset(self, state: ScheduleState) -> None:
        "Reset the objective's internal state."

    def on_task_started(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handles the objective change when a task starts its execution."

    def on_task_paused(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handles the objective change when a task is interrupted."

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        "Handles the objective change when a task ends its execution."

    def get_current(self, state: ScheduleState) -> float:
        """
        Get the current value of the objective function. This is useful for checking
        the performance of the scheduling algorithm along the episode.
        """
        return 0.0

    def __call__(self, state: ScheduleState) -> float:
        "Call the objective function to get the current value."
        return self.get_current(state)

    def get_entry(self) -> str:
        "Produce the γ entry for the constraint."
        return ""


class RegularObjective(Objective):
    """
    A regular objective is non-decreasing with respect to completion times.
    Any objective that depends solely on completion times is regular.
    """

    def __init__(
        self, minimize: bool = True
    ) -> None:
        super().__init__(minimize)

    @property
    def regular(self) -> bool:
        return True

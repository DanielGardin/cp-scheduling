from typing import NoReturn
from mypy_extensions import mypyc_attr

from cpscheduler.environment.state.events import Event
from cpscheduler.environment.state import ScheduleState

constraints: dict[str, type["Constraint"]] = {}


@mypyc_attr(allow_interpreted_subclasses=True)
class Constraint:
    """
    Base class for all constraints in the scheduling environment.
    This class provides a common interface for any piece in the scheduling environment that
    interacts with the tasks by limiting when they can be executed, how they are assigned to
    machines, etc.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        constraints[cls.__name__] = cls

    def initialize(self, state: ScheduleState) -> None:
        "Initialize the constraint with the scheduling state."

    def reset(self, state: ScheduleState) -> None:
        "Reset the constraint to its initial state."

    def propagate(self, event: Event, state: ScheduleState) -> None:
        "Given a bound change, propagate the constraint to other tasks."

    def get_entry(self) -> str:
        "Produce the β entry for the constraint."
        return ""


class PassiveConstraint(Constraint):
    """
    Passive constraints are compile-time constraints on the instance and do not interact with
    events during the scheduling process.
    They are used to provide task information and to set up the initial state for the scheduler.
    """

    def propagate(self, event: Event, state: ScheduleState) -> NoReturn:
        "Passive constraint does not propagate any changes."
        raise RuntimeError("Passive constraint does not propagate any changes.")

    def reset(self, state: ScheduleState) -> NoReturn:
        "Passive constraint does not reset any state."
        raise RuntimeError("Passive constraint does not reset any state.")

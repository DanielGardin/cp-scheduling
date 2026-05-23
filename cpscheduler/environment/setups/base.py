from mypy_extensions import mypyc_attr

from cpscheduler.environment.component import Component
from cpscheduler.environment.constraints import Constraint
from cpscheduler.environment.instance import ProblemInstance

setups: dict[str, type["ScheduleSetup"]] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class ScheduleSetup(Component):
    """
    Base class for scheduling setups. It defines the common interface for all scheduling setups
    and provides methods to parse process times, set tasks, and setup constraints.
    """

    def __init_subclass__(cls) -> None:
        name = cls.__name__

        if not name.startswith("_"):
            setups[name] = cls

    @property
    def n_machines(self) -> int:
        "Return the number of machines after the instance is loaded."
        return 0

    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        "Build the constraints for that setup."
        return ()

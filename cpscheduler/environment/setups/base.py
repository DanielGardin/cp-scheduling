"""Base class for scheduling setups."""

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.component import Component
from cpscheduler.environment.constraints import Constraint

setups: dict[str, type["ScheduleSetup"]] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class ScheduleSetup(Component):
    """Base class for scheduling setups.

    The setup component is responsible for defining the basic constraints of the
    scheduling problem, such as precedence constraints, resource constraints, etc.
    Each setup can be associated with a specific problem instance, and can define
    how to build the constraints for that instance.

    Subclasses of ScheduleSetup should implement the `setup_constraints` method to
    define the specific constraints for that setup.
    The `n_machines` property can be overridden to indicate the number of machines
    in the problem, if applicable.

    """

    @override
    def __init_subclass__(cls) -> None:
        name = cls.__name__

        if not name.startswith("_"):
            setups[name] = cls

    @property
    def n_machines(self) -> int:
        """Return the number of machines after the instance is loaded.

        If the number of machines is not fixed by the setup, return 0.
        """
        return 0

    def setup_constraints(self) -> tuple[Constraint, ...]:
        """Export the constraints defined by the setup.

        These constraints are intrinsic to the setup and are often
        initialized with the problem instance during `setup.initialize()`.
        """
        return ()

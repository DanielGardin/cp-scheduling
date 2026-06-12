"""Base class for scheduling setups."""

from typing import override

from mypy_extensions import mypyc_attr

from cpscheduler.environment.component import Component
from cpscheduler.environment.constraints import Constraint
from cpscheduler.environment.instance import ProblemInstance

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

    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        """Build the constraint objects to be included in the enviornment due to the setup.

        Parameters
        ----------
        instance: ProblemInstance
            The problem instance for which to build the constraints.

        """
        return ()

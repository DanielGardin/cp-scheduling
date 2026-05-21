from typing import TYPE_CHECKING
from collections.abc import Sequence

from cpscheduler.environment.constants import EzPickle

from mypy_extensions import mypyc_attr

if TYPE_CHECKING:
    from cpscheduler.environment.instance import ProblemInstance, Feature
    from cpscheduler.environment.state import ScheduleState

@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Component(EzPickle):
    """Abstract component of the scheduling environment.
    """

    def get_features(self) -> Sequence["Feature"]:
        """Return the list of features required in the problem instance.
        """
        return []

    def initialize(self, instance: "ProblemInstance") -> None:
        """Initialize the internal state of the component after the instance has  
        been loaded to the environment.

        Note that any change in the component's configuration after the 
        initialization will not take place until the next instance.
        """
    
    def reset(self, state: "ScheduleState") -> None:
        """Reset the component to its initial state and apply any changes in 
        the state.
        """

    def get_entry(self) -> str:
        "Produce the Graham entry for the component."
        return self.get_general_entry()

    @classmethod
    def get_general_entry(cls) -> str:
        return ""

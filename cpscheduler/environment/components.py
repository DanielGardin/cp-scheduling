from typing import Any

from cpscheduler.environment.constants import EzPickle

from cpscheduler.environment.instance import ProblemInstance, Feature
from cpscheduler.environment.state import ScheduleState

class Component(EzPickle):
    """Abstract component of the scheduling environment.
    """

    def get_features(self) -> list[Feature[Any]]:
        """Return the list of features required in the problem instance.
        """
        return []

    def initialize(self, instance: ProblemInstance) -> None:
        """Initialize the internal state of the component after the instance has  
        been loaded to the environment.

        Note that any change in the component's configuration after the 
        initialization will not take place until the next instance.
        """
    
    def reset(self, state: ScheduleState) -> None:
        """Reset the component to its initial state and apply any changes in 
        the state.
        """

    def get_entry(self) -> str:
        "Produce the Graham entry for the component."
        return ""

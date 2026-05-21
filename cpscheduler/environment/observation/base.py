from copy import deepcopy
from typing import Any, Generic
from typing_extensions import TypeVar

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle, MachineID, TaskID
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.state import ScheduleState

Serialized_Obs = TypeVar("Serialized_Obs", default=Any)

@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Observation(EzPickle, Generic[Serialized_Obs]):
    """Abstract observation contract for scheduling environments."""

    def initialize(self, instance: ProblemInstance) -> None:
        """Initialize the observation with the scheduling instance."""

    def update(self, state: ScheduleState) -> None:
        """Update the observation from the current stable scheduling state.
        
        This function is called immediatelly before the observation is returned 
        in the `step` and `reset` methods.
        Consider this method as a importer of the most recent 
        """

    def on_task_started(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle a task start event."""

    def on_task_paused(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle a task pause event."""

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle a task completion event."""

    def on_task_machine_infeasible(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        """Handle a task-machine infeasibility event."""

    def serialize(self) -> Serialized_Obs:
        """Return a serialized representation of the observation.
        
        Important: this method may not provide a copy of the observation, only
        a serialized version of the observation buffer inside the class.
        If you need to actual copies of the current observation, please refer 
        to `snapshot`.
        
        """
        raise NotImplementedError(
            f"serialize() was not implemented for {type(self).__name__}."
        )

    def snapshot(self) -> Serialized_Obs:
        """Return a serialized copy of the observation.
        
        By default, it uses `deepcopy` to generate an observation snapshot, but 
        this logic can be changed directly when required.
        """
        return deepcopy(self.serialize())

"""General Component class for scheduling components.

This module defines the Component base class, which will define the complete
life-cycle of each MDP parameter inside the environment.

We consider a component one of the following scheduling objects:
- Setups
- Constraints
- Objectives

Together, they compose the Graham entry (via `get_entry`), and define:
- How an instance of the problem is composed (via `get_features`).
- How many machines exist in the environment.
- What are the operational constraints.
- What is the scheduler goal.

"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle

if TYPE_CHECKING:
    from cpscheduler.environment.instance import Feature, ProblemInstance
    from cpscheduler.environment.state import ScheduleState


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Component(EzPickle):
    """Base class for environment components (setups, constraints, objectives).

    Components follow a three-phase lifecycle:
    1. **Configuration** (UNLOADED): Components are instantiated and configured.
    2. **Initialization** (LOADED): After instance loading, `initialize()` is called once.
    3. **Simulation** (RUNNING): `reset()` is called at each `env.reset()`, then the
       component processes state events during `step()`.

    Subclasses may provide features, maintain internal simulation state, or react to
    schedule transitions during execution.

    """

    def get_features(self) -> Sequence["Feature"]:
        """Return the list of features required or provided by this component.

        Features must be registered with the problem instance before `initialize()` is called.
        Typically called during `SchedulingEnv.__init__()` and after `load_instance()`.

        Returns
        -------
        Sequence[Feature]
            List of Feature objects (may be empty).

        """
        return []

    def initialize(self, instance: "ProblemInstance") -> None:
        """Initialize component state after the instance has been fully loaded.

        Called once per instance load, after all features are registered and instance
        data is ready. Any component configuration changes made after this call will
        not take effect until a new instance is loaded.

        Parameters
        ----------
        instance : ProblemInstance
            The loaded problem instance with features registered.

        """

    def reset(self, state: "ScheduleState") -> None:
        """Reset the component to initial simulation state.

        Called at the start of each episode (in `env.reset()`), after the schedule state
        has been cleared and before the first `step()` call. Use this to reset any
        internal counters, timers, or derived data.

        Parameters
        ----------
        state : ScheduleState
            The (freshly cleared) schedule state to operate on.

        """

    def get_entry(self) -> str:
        """Produce the Graham entry for the instantiated component."""
        return self.get_general_entry()

    @classmethod
    def get_general_entry(cls) -> str:
        """Produce the Graham entry for the component before instantiation."""
        return ""

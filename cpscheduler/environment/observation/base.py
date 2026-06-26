"""Base observation class for scheduling environments."""

from copy import deepcopy
from typing import Any, Generic

from mypy_extensions import mypyc_attr
from typing_extensions import TypeVar

from cpscheduler.environment.constants import EzPickle, MachineID, TaskID
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.specs import ObservationSpec
from cpscheduler.environment.state import ScheduleState

Serialized_Obs = TypeVar("Serialized_Obs", default=Any)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Observation(EzPickle, Generic[Serialized_Obs]):
    """Abstract observation contract for scheduling environments."""

    fingerprint: int
    n_tasks: int
    n_jobs: int
    n_machines: int
    symbols: dict[str, int]
    _default_symbols: dict[str, int]

    def __init__(
        self,
        n_tasks: int | None = None,
        n_machines: int | None = None,
        n_jobs: int | None = None,
        **symbols: int,
    ):
        """Initialize an Observation.

        Observations can be initialized with expected symbol values, which can
        be used to have a complete observation spec before any instance has
        been loaded.

        If the inferred symbols do not match the expectations, an error is raised
        during instance loading.

        By default, no symbol has an expected value.

        Parameters
        ----------
        n_tasks: int | None
            Expected number of tasks.

        n_machines: int | None
            Expected number of machines.

        n_jobs: int | None
            Expected number of jobs.
            If n_tasks is specified, but not n_jobs, it is supposed that
            n_jobs = n_tasks.

        **symbols: int
            Additional symbols with expected values.

        """
        self._default_symbols = {}

        if n_tasks is not None:
            self._default_symbols["n_tasks"] = n_tasks

            if n_jobs is None:
                n_jobs = n_tasks

        if n_machines is not None:
            self._default_symbols["n_machines"] = n_machines

        if n_jobs is not None:
            self._default_symbols["n_jobs"] = n_jobs

        self._default_symbols.update(symbols)
        self.symbols = self._default_symbols.copy()

        self.n_tasks = n_tasks or 0
        self.n_machines = n_machines or 0
        self.n_jobs = n_jobs or 0

    def initialize(self, instance: ProblemInstance) -> None:
        """Initialize the observation with the scheduling instance."""
        concrete_symbols = instance.symbol_values

        for symbol, value in self._default_symbols.items():
            if symbol not in concrete_symbols:
                raise KeyError(
                    f"Observation expected symbol '{symbol}', which hasn't been "
                    "defined by the problem."
                )

            if value != concrete_symbols[symbol]:
                raise ValueError(
                    f"Observation expected {symbol}={value}, but the loaded "
                    f"instance has {symbol}={concrete_symbols[symbol]}."
                )

        self.fingerprint = instance.fingerprint

        self.n_tasks = instance.n_tasks
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
        self.symbols = instance.symbol_values

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

    def get_spec(self) -> ObservationSpec:
        """Return the specification of this observation structure.

        Describes the layout, types, and metadata of the observation
        without requiring serialized data.
        """
        raise NotImplementedError(
            f"get_spec() was not implemented for {type(self).__name__}."
        )

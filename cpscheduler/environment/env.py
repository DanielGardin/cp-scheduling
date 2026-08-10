"""Discrete-event environment for constraint-based scheduling problems.

The environment composes a machine setup (alpha), constraints (beta), and an objective (gamma)
under a Graham-style scheduling formulation.
It exposes a Gymnasium-like interface over an event-driven simulation kernel
with constraint propagation.
"""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from typing_extensions import TypeVar, assert_never

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.constraints import Constraint, PassiveConstraint
from cpscheduler.environment.des import (
    ActionType,
    Schedule,
    is_single_action,
    parse_instruction,
)
from cpscheduler.environment.instance import FeatureMetadata, ProblemInstance
from cpscheduler.environment.objectives import Objective
from cpscheduler.environment.observation import DefaultObservation, Observation
from cpscheduler.environment.render import Renderer
from cpscheduler.environment.setups import ScheduleSetup
from cpscheduler.environment.specs import ObservationSpec
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.state.events import VarField
from cpscheduler.environment.tracer import Tracer
from cpscheduler.environment.utils import (
    InfoType,
    Instance_T,
    InstanceGenerator,
    InstanceTypes,
    Metric,
    Options,
    ensure_iterable,
)

if TYPE_CHECKING:
    from cpscheduler.environment.component import Component

# Event fields and kinds

EnvStatusType = Literal[0, 1, 2]

# No instance loaded. Constraints and Objectives can be added
UNLOADED: Literal[0] = 0

# Instance loaded and components initialized, configuration frozen
LOADED: Literal[1] = 1

# State globally consistent, instance frozen
RUNNING: Literal[2] = 2

ASSIGNMENT = VarField.ASSIGNMENT
START_LB = VarField.START_LB
START_UB = VarField.START_UB
END_LB = VarField.END_LB
END_UB = VarField.END_UB
PRESENCE = VarField.PRESENCE
ABSENCE = VarField.ABSENCE
MACHINE_INFEASIBLE = VarField.MACHINE_INFEASIBLE
STATE_INFEASIBLE = VarField.STATE_INFEASIBLE


def _join_info(info: InfoType, key: str, new_item: InfoType | Any) -> None:
    if isinstance(new_item, Mapping):
        info.update(
            {f"{key}.{subkey}": value for subkey, value in new_item.items()}
        )

    else:
        info[key] = new_item


ObsT_co = TypeVar(
    "ObsT_co", bound=Observation, default=Observation[Any], covariant=True
)


class SchedulingEnv(EzPickle, Generic[ObsT_co]):
    """Discrete-event environment for scheduling problems with constraint propagation.

    SchedulingEnv combines a machine setup (alpha), constraints (beta), and objectives (gamma) following
    the Graham scheduling notation. The environment enforces feasibility via constraint
    propagation and provides a Gymnasium-like interface for RL integration.

    State Machine
    -------------
    UNLOADED
        No instance loaded.

    LOADED
        Instance loaded and components initialized.

    RUNNING
        Episode active, simulation advances via step().

    Simulation Loop (in `step()`)
    ----------------------------
    1. Accept action from policy (task assignment or batch).
    2. Advance simulation time to next event horizon determined by schedule or state.
    3. Process events (constraint propagation until fixed-point).
    4. Update runtime (callbacks for task start/completion).
    5. Return observation, reward, terminated, and info.

    """

    # Environment static variables
    setup: ScheduleSetup
    constraints: tuple[Constraint, ...]
    objective: Objective
    observation: ObsT_co
    observation_spec: ObservationSpec

    metrics: dict[str, Metric]
    tracers: tuple[Tracer, ...]

    renderer: Renderer

    # Environment dynamic variables
    setup_constraints: tuple[Constraint, ...]
    _all_constraints: tuple[Constraint, ...]
    instance: ProblemInstance
    instance_generator: InstanceGenerator | None
    state: ScheduleState
    schedule: Schedule

    # Helper variables
    _prev_obj_value: float
    event_count: int

    _status: EnvStatusType

    # FUTURE: Mypyc issue https://github.com/mypyc/mypyc/issues/961
    # The trick used here produces a false-positive when type checking:
    # env:  SchedulingEnv[OtherObservation] = SchedulingEnv()
    # @overload
    # def __init__(
    #     self: "SchedulingEnv[DefaultObservation]",
    #     machine_setup: ScheduleSetup | None = None,
    #     constraints: Iterable[Constraint] | None = None,
    #     objective: Objective | None = None,
    #     observation: None = None,
    #     instance: InstanceTypes | InstanceGenerator | None = None,
    #     metrics: Mapping[str, Metric] | None = None,
    #     tracers: Iterable[Tracer] | None = None,
    #     render_mode: Renderer | str | None = None,
    #     debug_mode: bool = False,
    # ) -> None: ...

    # @overload
    # def __init__(
    #     self,
    #     machine_setup: ScheduleSetup | None = None,
    #     constraints: Iterable[Constraint] | None = None,
    #     objective: Objective | None = None,
    #     observation: ObsT_co | None = None,
    #     instance: InstanceTypes | InstanceGenerator | None = None,
    #     metrics: Mapping[str, Metric] | None = None,
    #     tracers: Iterable[Tracer] | None = None,
    #     render_mode: Renderer | str | None = None,
    #     debug_mode: bool = False,
    # ) -> None: ...

    def __init__(
        self,
        machine_setup: ScheduleSetup | None = None,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        observation: ObsT_co | None = None,
        instance: InstanceTypes | InstanceGenerator | None = None,
        metrics: Mapping[str, Metric] | None = None,
        tracers: Iterable[Tracer] | None = None,
        render_mode: Renderer | str | None = None,
        debug_mode: bool = False,
    ):
        """Initialize the scheduling environment.

        Defines the parameterized environment, composing machine_setup,
        constraints and objective.

        If `instance` is an `InstanceGenerator`, it is stored for deferred
        sampling after each `reset()` call.
        If it is concrete instance data, `load_instance()` is called immediately.

        Parameters
        ----------
        machine_setup : ScheduleSetup, optional
            Alpha component in Graham notation. Defines machine topology and
            availability. Defaults to a bare `ScheduleSetup` (no machines).

        constraints : Iterable[Constraint], optional
            Beta components. Applied during constraint propagation.
            Defaults to an empty tuple.

        objective : Objective, optional
            Gamma component. Defines the reward signal. Defaults to a no-op
            `Objective`.

        observation : ObsT_co, optional
            Observation class used to build RL observations. Defaults to
            `DefaultObservation`.

        instance : InstanceTypes or InstanceGenerator, optional
            Either concrete instance data or a generator stored for lazy sampling.

        metrics : Mapping[str, Metric], optional
            Performance metrics to be added to the info dictionary.

        tracers : Iterable[Tracer], optional
            Tracers to monitor internal state before each decision step.

        render_mode : Renderer or str, optional
            Renderer instance or mode string. Defaults to a no-op renderer.

        debug_mode : bool, optional
            Enables additional runtime assertions during simulation.
            Defaults to False.

        """
        self._status = UNLOADED

        if machine_setup is None:
            machine_setup = ScheduleSetup()

        if constraints is None:
            constraints = ()

        if objective is None:
            objective = Objective()

        if observation is None:
            observation = cast("ObsT_co", DefaultObservation())

        setup_constraints = machine_setup.setup_constraints()

        problem_instance = ProblemInstance(debug_mode)

        component: Component
        for component in [
            machine_setup,
            *constraints,
            *setup_constraints,
            objective,
        ]:
            for feature in component.get_features():
                problem_instance.register(feature)

        if "n_machines" not in observation.symbols:
            observation.symbols["n_machines"] = machine_setup.n_machines

        for symbol in problem_instance.symbols:
            if symbol not in observation.symbols:
                observation.symbols[symbol] = 0

        self.setup = machine_setup
        self.constraints = tuple(constraints)
        self.objective = objective
        self.observation = observation
        self.observation_spec = observation.compile(problem_instance)

        self.metrics = dict(metrics) if metrics is not None else {}
        self.tracers = tuple(tracers) if tracers is not None else ()

        self.setup_constraints = setup_constraints
        self._all_constraints = tuple(
            constraint
            for constraint in self.constraints + setup_constraints
            if not isinstance(constraint, PassiveConstraint)
        )
        self.instance = problem_instance

        self.schedule = Schedule()

        self.instance_generator = None
        if isinstance(instance, InstanceGenerator):
            self.instance_generator = instance

        elif instance is not None:
            self.load_instance(*ensure_iterable(instance))

        self.renderer = (
            render_mode
            if isinstance(render_mode, Renderer)
            else Renderer.get_renderer(render_mode)
        )

    @property
    def fingerprint(self) -> int:
        """Return the instance fingerprint of the current loaded instance."""
        return self.instance.fingerprint

    @property
    def raw_instance(self) -> dict[str, Any]:
        """Return raw instance data."""
        return self.instance.original_instance

    @property
    def loaded(self) -> bool:
        """Indicates whether an instance has been loaded and the environment is initialized."""
        return self._status != UNLOADED

    @property
    def running(self) -> bool:
        """Indicates whether the environment is currently running an episode."""
        return self._status == RUNNING

    @property
    def all_constraints(self) -> tuple[Constraint, ...]:
        """Access all active constraints used during propagation."""
        return self._all_constraints

    def __repr__(self) -> str:
        """Return a string representation of the environment's configuration and state."""
        entry = self.get_entry()

        if self._status == UNLOADED:
            return f"SchedulingEnv({entry}, n_tasks=0)"

        state = self.state
        n_tasks = state.n_tasks

        if self._status == LOADED:
            return f"SchedulingEnv({entry}, n_tasks={n_tasks})"

        if state.infeasible:
            return f"SchedulingEnv({entry}, n_tasks={n_tasks}, infeasible=True)"

        obj_value = self.objective.get_current(state)

        return (
            f"SchedulingEnv({entry}, n_tasks={n_tasks}, objective={obj_value})"
        )

    # Environment configuration public methods
    def reset_instance(self) -> None:
        """Unload the current instance and restore configuration mutability.

        Returns the environment to UNLOADED state, allowing setup, constraints,
        and objective modifications.
        """
        self.instance.reset()
        self._status = UNLOADED

    def required_features(
        self, show_optional: bool = False
    ) -> dict[str, FeatureMetadata]:
        """Return a dictionary of all required features from the setup, constraints, and objective."""
        return self.instance.required_features(show_optional)

    def set_generator(self, instance_generator: InstanceGenerator) -> None:
        """Set the instance generator.

        The instance generator will sample a new instance for every reset call
        """
        self.instance_generator = instance_generator

    def load_instance(self, *instances: Instance_T) -> None:
        """Load a scheduling instance and initialize the environment.

        Prepares the environment for simulation by loading instance data,
        validating constraints, and propagating domain bounds.

        Parameters
        ----------
        *instances : InstanceTypes
            One or more instance data objects to load.
            If multiple instances are provided, they are merged.
            Allows for heterogeneous instance data sources.

        Raises
        ------
        ValueError
            If setup constraints produce invalid features.

        RuntimeError
            If constraint propagation detects initial infeasibility.

        """
        self.reset_instance()

        problem_instance = self.instance

        problem_instance.initialize(instances, self.setup)
        self.setup.initialize(problem_instance)
        problem_instance.finalize()

        component: Component
        for component in [
            *self.setup_constraints,
            *self.constraints,
            self.objective,
        ]:
            component.initialize(problem_instance)

        self.observation.initialize(problem_instance)

        for tracer in self.tracers:
            tracer.initialize(problem_instance)

        self.state = ScheduleState(problem_instance)
        self._status = LOADED

    def add_metric(self, name: str, metric: Metric) -> None:
        """Add a metric to the environment."""
        self.metrics[name] = metric

    def clear_metrics(self) -> None:
        """Clear all metrics from the environment."""
        self.metrics.clear()

    def get_entry(self) -> str:
        """Get a string representation of the environment's configuration."""
        alpha = self.setup.get_entry()

        beta = ",".join(
            [constraint.get_entry() for constraint in self.constraints]
        )

        gamma = self.objective.get_entry()

        return f"{alpha}|{beta}|{gamma}"

    # Environment state retrieval methods
    def get_info(self) -> InfoType:
        """Retrieve additional information about the environment."""
        info = {
            "objective_value": self._prev_obj_value,
            "event_count": self.event_count,
            "infeasible": self.state.infeasible,
        }

        for tracer in self.tracers:
            tracer_info = tracer.export()

            if tracer_info is not None:
                _join_info(info, tracer.tracer_name, tracer_info)

        for metric_name, metric in self.metrics.items():
            metric_value = metric(self.state)
            _join_info(info, metric_name, metric_value)

        return info

    # Environment internal methods for simulation

    # TODO: Add a meaningful error message when the action is not parseable.
    def schedule_action(self, action: ActionType) -> None:
        """Parse action and add instruction(s) to the event schedule.

        Handles both single actions and batch actions, converting them into
        instructions for the simulation.
        """
        if action is None:
            return

        state = self.state

        if is_single_action(action):
            instruction, time, priority = parse_instruction(action)
            self.schedule.add_event(instruction, state, time, priority)

            return

        for instruction_args in action:
            instruction, time, priority = parse_instruction(instruction_args)
            self.schedule.add_event(instruction, state, time, priority)

    # def advance_clock(self) -> bool:
    #     """Advance simulation time to the next event horizon.

    #     Determines the next time step based on scheduled instructions or state,
    #     updates all constraints, and triggers propagation.
    #     Halts when the schedule is empty, awaiting the next policy action.

    #     The next time is determined by the following logic:
    #     - If the schedule has pending instructions, advance to the next instruction time.
    #     - If the schedule is empty but there are unlocked tasks, advance to the earliest start_lb
    #         among unlocked tasks.
    #     - If the schedule is empty and no unlocked tasks, advance to the last completion time.

    #     This logic ensures that the simulation advances to the next decision
    #     point where the state may change, either due to scheduled events or
    #     lower bounds for next possible instructions.

    #     Returns
    #     -------
    #     bool
    #         True if schedule is not empty (more events to process).
    #         False if inconsistency detected or schedule is empty.

    #     """
    #     schedule = self.schedule
    #     state = self.state

    #     empty_schedule = schedule.is_empty()

    #     if not empty_schedule:
    #         next_time = schedule.next_time()

    #     elif state.runtime.unlocked_tasks:
    #         next_time = state.get_next_start_lb()

    #     else:
    #         next_time = state.get_last_completion_time()

    #     if next_time > state.time:
    #         self.state.advance_time_(next_time)

    #         for constraint in self._all_constraints:
    #             constraint.on_time_update(next_time, self.state)

    #         self.propagate()

    #     return not schedule.is_empty()

    def propagate(self) -> None:
        """Execute constraint propagation to fixed-point.

        Processes all domain events in the queue, invoking constraint callbacks
        (assignment, bounds updates, presence/absence, etc.) until the queue is
        exhausted or infeasibility is detected.

        Returns
        -------
        bool
            True if propagation reached fixed-point.
            False if state is infeasible.

        """
        state = self.state
        event_queue = state.domain_event_queue
        constraints = self._all_constraints

        # FUTURE: For performance, consider subscribing constraints such that,
        # constraint.fields() -> Sequence[VarFieldType]
        # then we cache which propagators are subscribed for a field, instead of
        # iterating over all constraints.

        task_ids = event_queue.task_ids
        fields = event_queue.fields
        machine_ids = event_queue.machine_ids

        idx = 0
        while idx < len(event_queue):
            task_id = task_ids[idx]
            field = fields[idx]
            machine_id = machine_ids[idx]

            if field == ASSIGNMENT:
                for constraint in constraints:
                    constraint.on_assignment(task_id, machine_id, state)

            elif field == START_LB:
                for constraint in constraints:
                    constraint.on_start_lb(task_id, machine_id, state)

            elif field == START_UB:
                for constraint in constraints:
                    constraint.on_start_ub(task_id, machine_id, state)

            elif field == END_LB:
                for constraint in constraints:
                    constraint.on_end_lb(task_id, machine_id, state)

            elif field == END_UB:
                for constraint in constraints:
                    constraint.on_end_ub(task_id, machine_id, state)

            elif field == PRESENCE:
                for constraint in constraints:
                    constraint.on_presence(task_id, state)

            elif field == ABSENCE:
                for constraint in constraints:
                    constraint.on_absence(task_id, state)

            elif field == MACHINE_INFEASIBLE:
                for constraint in constraints:
                    constraint.on_infeasibility(task_id, machine_id, state)

            # FUTURE: This should be resolved when the event is created to allow
            # discovering the causal effect that lead to infeasibility.
            # In the current version, the event queue is not cleared to make
            # discovery easier, but it must not be the final behavior.
            elif field == STATE_INFEASIBLE:
                assert state.infeasible, (
                    "STATE_INFEASIBLE event produced erroneously."
                )
                self.event_count += idx + 1
                return

            else:
                assert_never(field)

            idx += 1

        self.event_count += idx
        event_queue.clear()

    # Environment API methods
    def reset(
        self, *, options: Options | None = None
    ) -> tuple[ObsT_co, InfoType]:
        """Reset the environment to its initial state for a new episode.

        Clears the schedule, resets all state variables, and performs initial
        constraint propagation. Optionally loads a new instance via generator.

        Parameters
        ----------
        options : dict, optional
            Configuration options including:
            - 'instance': Load a specific instance.
            - 'instance_generator': Replace the instance generator.
            - 'seed': Seed for the generator.

        Returns
        -------
        observation : ObsT_co
            Initial observation.

        info : dict[str, Any]
            Environment info (time, objective value, event count, etc.).

        Raises
        ------
        ValueError
            If no instance is loaded.

        RuntimeError
            If propagation detects initial inconsistency.

        Note
        ----
        When resetting the environment will resolve which instance will be
        simulated by the following order of priority, selecting the first
        one encountered:

        1. The instance given in option by the "instance" key.
        2. A generated instance from a generator given by "instance_generator" key.
        3. A generated instance from the internal generator.
        4. The instance that was already previously loaded.

        """
        options = options or {}

        if "instance" in options:
            self.load_instance(*ensure_iterable(options["instance"]))

        else:
            generator = options.get("instance_generator")
            if isinstance(generator, InstanceGenerator):
                self.instance_generator = generator

            if self.instance_generator is not None:
                sample = self.instance_generator.sample(options.get("seed"))

                self.load_instance(*ensure_iterable(sample))

        if self._status == UNLOADED:
            raise ValueError(
                "Environment has not been loaded with an instance. "
                "Please call reset(options={'instance':<instance>}) or "
                "load_instance(<instance>) before resetting."
            )

        state = self.state

        self.schedule.reset()
        state.reset()

        for tracer in self.tracers:
            tracer.reset(state)

        for component in [self.setup, *self._all_constraints, self.objective]:
            component.reset(state)

        self.event_count = 0
        self.propagate()

        if state.infeasible:
            raise RuntimeError(
                "A stale state was produced after reset. Perhaps produced by "
                "contradictory constraints included in your schedule problem."
            )

        observation = self.observation

        observation.update(state)
        self._prev_obj_value = self.objective.get_current(state)  # Cold start

        self._status = RUNNING

        return observation, self.get_info()

    def step(
        self, action: ActionType = None
    ) -> tuple[ObsT_co, float, bool, bool, InfoType]:
        """Execute one simulation step.

        Schedules the action, advances time, processes instructions, and updates
        the observation and reward.
        Note that a step in the environment can correspond to many unit time
        steps in the environment's clock.

        A step here is defined as the state advancement where one of the
        following conditions is met first:
        - The instruction queue is executed entirely.
        - A terminal state is reached.
        - A infeasibility is detected (a bad action caused a contradiction)

        Parameters
        ----------
        action : ActionType, optional
            Task assignment(s) or None.

        Returns
        -------
        observation : ObsT_co
            Current observation.

        reward : float
            Signed reward (Delta objective value).

        terminated : bool
            Whether all tasks are completed.

        truncated : bool
            Whether the episode reached an infeasible state.

        info : dict[str, Any]
            Environment info (time, objective, event count, etc.).

        Raises
        ------
        RuntimeError
            If environment is not in RUNNING state.
            Ensure you called `reset()` before `step()`.

        """
        state = self.state

        if self._status != RUNNING:
            if self._status == UNLOADED:
                raise RuntimeError(
                    "Environment was not reset after loading an instance, or "
                    "wasn't loaded. Please either call "
                    "reset(options={'instance':<instance>}) or "
                    "load_instance(<instance>), then reset()."
                )

            raise RuntimeError(
                "An instance was loaded, but the environment has not been "
                "initialized. Please call reset() before step()."
            )

        self.schedule_action(action)

        schedule = self.schedule

        while not state.is_terminal():
            for event in schedule.instruction_queue(state):
                for tracer in self.tracers:
                    tracer.step(state, event)

                event.process(state, schedule)

                self.propagate()

                if state.infeasible:
                    break

        # Gymnasium-like step return

        self.observation.update(state)

        obj_value = self.objective.get_current(state)
        reward = obj_value - self._prev_obj_value
        self._prev_obj_value = obj_value

        if self.objective.minimize:
            reward = -reward

        truncated = state.infeasible
        terminal = state.is_terminal()
        info = self.get_info()

        return self.observation, reward, terminal, truncated, info

    def render(self) -> None:
        """Render the current environment state.

        Raises
        ------
        RuntimeError
            If environment is not in RUNNING state.

        """
        if self._status != RUNNING:
            raise RuntimeError(
                "Cannot render an environment during configuration."
            )

        self.renderer.render(self.state)

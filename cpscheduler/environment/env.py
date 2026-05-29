"""
env.py

This component defines the SchedulingEnv class, which is a custom environment for generic
scheduling problems.
It is designed to be modular and extensible, allowing users to define their own scheduling
problems by specifying the machine setup, constraints, objectives, and instances.

The environment is based on the OpenAI Gym interface, making it compatible with various
reinforcement learning libraries. It provides methods for resetting the environment, taking
steps, rendering the environment, and exporting the scheduling model.
"""

from collections.abc import Iterable, Mapping
from typing import Generic, Literal, cast

from typing_extensions import TypeVar, assert_never

from cpscheduler.environment.component import Component
from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.constraints import Constraint, PassiveConstraint
from cpscheduler.environment.des import (
    ActionType,
    Schedule,
    is_single_action,
    parse_instruction,
)
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.objectives import Objective
from cpscheduler.environment.observation import DefaultObservation, Observation
from cpscheduler.environment.render import Renderer
from cpscheduler.environment.setups import ScheduleSetup
from cpscheduler.environment.specs.feature_spec import FeatureSpec
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.state.events import RuntimeEventKind, VarField
from cpscheduler.environment.utils.protocols import (
    InfoType,
    InstanceGenerator,
    InstanceTypes,
    Metric,
    Options,
)

# Event fields and kinds

EnvStatusType = Literal[0, 1, 2]

UNLOADED: Literal[0] = 0
"No instance loaded. Constraints and Objectives can be added."

LOADED: Literal[1] = 1
"Instance loaded and components initialized, configuration frozen"

RUNNING: Literal[2] = 2
"State globally consistent, instance frozen."

ASSIGNMENT = VarField.ASSIGNMENT
START_LB = VarField.START_LB
START_UB = VarField.START_UB
END_LB = VarField.END_LB
END_UB = VarField.END_UB
PRESENCE = VarField.PRESENCE
ABSENCE = VarField.ABSENCE
MACHINE_INFEASIBLE = VarField.MACHINE_INFEASIBLE
PAUSE = VarField.PAUSE
BOUNDS_RESET = VarField.BOUNDS_RESET
STATE_INFEASIBLE = VarField.STATE_INFEASIBLE

TASK_STARTED = RuntimeEventKind.TASK_STARTED
TASK_PAUSED = RuntimeEventKind.TASK_PAUSED
TASK_COMPLETED = RuntimeEventKind.TASK_COMPLETED
TASK_MACHINE_INFEASIBLE = RuntimeEventKind.TASK_MACHINE_INFEASIBLE

ObsT_co = TypeVar(
    "ObsT_co", bound=Observation, default=DefaultObservation, covariant=True
)


class SchedulingEnv(EzPickle, Generic[ObsT_co]):
    """
    SchedulingEnv is a custom environment for generic scheduling problems. It is designed to be
    modular and extensible, allowing users to define their own scheduling problems by specifying
    the machine setup, constraints, objectives, and instances.

    The environment is based on the OpenAI Gym interface, making it compatible with various
    reinforcement learning libraries.
    """

    # Environment static variables
    setup: ScheduleSetup
    constraints: list[Constraint]
    setup_constraints: tuple[Constraint, ...]
    objective: Objective
    instance_generator: InstanceGenerator | None
    _all_constraints: tuple[Constraint, ...]
    _observation: ObsT_co

    metrics: dict[str, Metric]

    renderer: Renderer

    # Environment dynamic variables
    _instance: ProblemInstance
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
        render_mode: Renderer | str | None = None,
        debug_mode: bool = False,
    ):
        self._status = UNLOADED

        problem_instance = ProblemInstance(debug_mode)

        if machine_setup is None:
            machine_setup = ScheduleSetup()

        if constraints is None:
            constraints = ()

        if objective is None:
            objective = Objective()

        if observation is None:
            observation = cast(ObsT_co, DefaultObservation())

        component: Component
        for component in [
            machine_setup,
            *constraints,
            objective,
        ]:
            for feature in component.get_features():
                problem_instance.register(feature)

        for feature in observation.get_features():
            problem_instance.register(feature)

        self._instance = problem_instance
        self.setup = machine_setup
        self.constraints = list(constraints)
        self.objective = objective
        self._observation = observation

        self.schedule = Schedule()

        self.setup_constraints = ()
        self._all_constraints = ()

        self.metrics = dict(metrics) if metrics is not None else {}

        self.instance_generator = None
        if isinstance(instance, InstanceGenerator):
            self.instance_generator = instance

        elif instance is not None:
            self.load_instance(instance)

        self.renderer = (
            render_mode
            if isinstance(render_mode, Renderer)
            else Renderer.get_renderer(render_mode)
        )

    @property
    def loaded(self) -> bool:
        return self._status != UNLOADED

    @property
    def running(self) -> bool:
        return self._status == RUNNING

    @property
    def observation(self) -> ObsT_co:
        return self._observation

    @property
    def all_constraints(self) -> tuple[Constraint, ...]:
        return self._all_constraints

    def __repr__(self) -> str:
        entry = self.get_entry()

        if self._status == UNLOADED:
            return f"SchedulingEnv({entry}, n_tasks=0)"

        state = self.state
        n_tasks = state.n_tasks

        if self._status == LOADED:
            return f"SchedulingEnv({entry}, n_tasks={n_tasks})"

        time = state.time

        if state.infeasible:
            return (
                f"SchedulingEnv({entry}, n_tasks={n_tasks}, "
                f"current_time={time}, infeasible=True)"
            )

        obj_value = self.objective.get_current(state)

        return (
            f"SchedulingEnv({entry}, n_tasks={n_tasks}, "
            f"current_time={time}, objective={obj_value})"
        )

    # Environment configuration public methods
    def add_constraint(self, constraint: Constraint) -> None:
        "Add a constraint to the environment."
        if self._status != UNLOADED:
            self.reset_instance()

        self.constraints.append(constraint)

        instance = self._instance
        for feature in constraint.get_features():
            instance.register(feature)

    def set_objective(self, objective: Objective) -> None:
        "Set the objective function for the environment."
        if self._status != UNLOADED:
            self.reset_instance()

        instance = self._instance
        for prev_feature in self.objective.get_features():
            instance.unregister(prev_feature)

        for feature in objective.get_features():
            instance.register(feature)

        self.objective = objective

    def reset_instance(self) -> None:
        """Remove the instance and allow the configuration of the environment to
        change.
        """
        instance = self._instance

        for constraint in self.setup_constraints:
            for feature in constraint.get_features():
                instance.unregister(feature)

        self.setup_constraints = ()
        self._all_constraints = ()

        instance.reset()
        self._status = UNLOADED

    def required_features(self) -> dict[str, FeatureSpec]:
        return self._instance.required_features()

    def load_instance(self, instance: InstanceTypes) -> None:
        """
        Set the instance data for the environment.

        Parameters:
            instance: InstanceTypes
                The instance data for the scheduling problem, can be a DataFrame or a dictionary.
        """
        self.reset_instance()

        problem_instance = self._instance

        problem_instance.initialize(instance, self.setup)
        self.setup.initialize(problem_instance)

        setup_constraints = self.setup.setup_constraints(problem_instance)
        for constraint in setup_constraints:
            for feature in constraint.get_features():
                if not feature.owner:
                    raise ValueError(
                        f"Setup '{self.setup}' produced a non-owner feature "
                        f"'{feature.name}' from setup constraint '{constraint}'. "
                        f"Setup constraints must be built entirely from the "
                        "instance, and not require any additional features."
                    )

                problem_instance.register(feature)

        self.setup_constraints = setup_constraints
        self._all_constraints = (
            *setup_constraints,
            *(
                constraint
                for constraint in self.constraints
                if not isinstance(constraint, PassiveConstraint)
            ),
        )

        component: Component
        for component in [
            *self.setup_constraints,
            *self.constraints,
            self.objective,
        ]:
            component.initialize(problem_instance)

        self._observation.initialize(problem_instance)

        problem_instance.validate_instance("SchedulingEnv.load_instance")

        self.state = ScheduleState(problem_instance)
        self._status = LOADED

    def add_metric(self, name: str, metric: Metric) -> None:
        "Add a metric to the environment."
        self.metrics[name] = metric

    def clear_metrics(self) -> None:
        "Clear all metrics from the environment."
        self.metrics.clear()

    def get_entry(self) -> str:
        "Get a string representation of the environment's configuration."
        alpha = self.setup.get_entry()

        beta = ",".join(
            [constraint.get_entry() for constraint in self.constraints]
        )

        gamma = self.objective.get_entry()

        return f"{alpha}|{beta}|{gamma}"

    # Environment state retrieval methods
    def get_info(self) -> InfoType:
        "Retrieve additional information about the environment."
        info = {
            "current_time": self.state.time,
            "objective_value": self._prev_obj_value,
            "event_count": self.event_count,
            "infeasible": self.state.infeasible,
        }

        for metric_name, metric in self.metrics.items():
            metric_value = metric(self.state)

            if isinstance(metric_value, Mapping):
                info.update(
                    {
                        f"{metric_name}_{key}": value
                        for key, value in metric_value.items()
                    }
                )

            else:
                info[metric_name] = metric_value

        return info

    # Environment internal methods for simulation
    def schedule_action(self, action: ActionType) -> None:
        "Consume an action by adding the corresponding instruction(s) to the schedule."
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

    def advance_clock(self) -> bool:
        schedule = self.schedule
        state = self.state

        empty_schedule = schedule.is_empty()

        if not empty_schedule:
            next_time = schedule.next_time()

        elif state.runtime.unlocked_tasks:
            next_time = state.get_next_start_lb()

        else:
            next_time = state.get_last_completion_time()

        if next_time > state.time:
            self.state.advance_time_(next_time)

            for constraint in self._all_constraints:
                constraint.on_time_update(next_time, self.state)

            consistent = self.propagate()

            if not consistent:
                return False

        # When the schedule is empty, the environment halts, expecting an action
        # from the policy.
        return not schedule.is_empty()

    def propagate(self) -> bool:
        """
        Propagate the new bounds through the constraints until a fixed-point is reached.

        Returns whether the propagation has run without producing an infeasible state.
        """

        state = self.state
        event_queue = state.domain_event_queue
        constraints = self._all_constraints

        # FUTURE: For performance, consider subscribing constraints such that,
        # constraint.fields() -> Sequence[VarFieldType]
        # then we cache which propagators are subscribed for a field, instead of
        # iterating over all constraints.

        idx = 0
        while idx < len(event_queue):
            event = event_queue[idx]
            task_id = event.task_id
            machine_id = event.machine_id
            field = event.field

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

            elif field == PAUSE:
                for constraint in constraints:
                    constraint.on_pause(task_id, machine_id, state)

            elif field == BOUNDS_RESET:
                for constraint in constraints:
                    constraint.on_bound_reset(task_id, state)

            # FUTURE: This should be resolved when the event is created to allow
            # discovering the causal effect that lead to infeasibility.
            # In the current version, the event queue is not cleared to make
            # discovery easier, but it must not be the final behavior.
            elif field == STATE_INFEASIBLE:
                assert state.infeasible, (
                    "STATE_INFEASIBLE event produced erroneously."
                )
                self.event_count += idx + 1
                return False

            else:
                assert_never(field)

            idx += 1

        self.event_count += idx
        event_queue.clear()

        return True

    def update_runtime(self) -> None:
        state = self.state
        runtime_event_queue = state.runtime_event_queue
        objective = self.objective
        observation = self._observation

        for event in runtime_event_queue:
            task_id = event.task_id
            kind = event.kind
            machine_id = event.machine_id

            if kind == TASK_STARTED:
                objective.on_task_started(task_id, machine_id, state)
                observation.on_task_started(task_id, machine_id, state)

            elif kind == TASK_PAUSED:
                objective.on_task_paused(task_id, machine_id, state)
                observation.on_task_paused(task_id, machine_id, state)

            elif kind == TASK_COMPLETED:
                objective.on_task_completed(task_id, machine_id, state)
                observation.on_task_completed(task_id, machine_id, state)

            elif kind == TASK_MACHINE_INFEASIBLE:
                objective.on_task_machine_infeasible(task_id, machine_id, state)
                observation.on_task_machine_infeasible(
                    task_id, machine_id, state
                )

            else:
                assert_never(kind)

        runtime_event_queue.clear()

    # Environment API methods
    def reset(
        self, *, options: Options | None = None
    ) -> tuple[ObsT_co, InfoType]:
        options = options or {}

        if "instance" in options:
            self.load_instance(options["instance"])

        else:
            generator = options.get("instance_generator")
            if isinstance(generator, InstanceGenerator):
                self.instance_generator = generator

            if self.instance_generator is not None:
                seed = options.get("seed")
                self.load_instance(self.instance_generator.sample(seed))

        if self._status == UNLOADED:
            raise ValueError(
                "Environment has not been loaded with an instance. "
                "Please call reset(options={'instance':<instance>}) or "
                "load_instance(<instance>) before resetting."
            )

        state = self.state

        self.schedule.reset()
        state.reset()

        for component in [self.setup, *self._all_constraints, self.objective]:
            component.reset(state)

        self.event_count = 0
        consistent = self.propagate()

        if not consistent:
            raise RuntimeError(
                "A stale state was produced after reset. Perhaps produced by "
                "contradictory constraints included in your schedule problem."
            )

        observation = self._observation

        observation.update(state)
        self._prev_obj_value = self.objective.get_current(state)  # Cold start

        self._status = RUNNING

        return observation, self.get_info()

    def step(
        self, action: ActionType = None
    ) -> tuple[ObsT_co, float, bool, bool, InfoType]:
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

        consistent = True
        while consistent and not state.is_terminal() and self.advance_clock():
            for event in schedule.instruction_queue(state):
                # After each instruction is processed, domains are updated until a fixed point.
                event.process(state, schedule)

                consistent = self.propagate()

                if not consistent:
                    break

        # Gymnasium-like step return

        self.update_runtime()
        self._observation.update(state)

        obj_value = self.objective.get_current(state)
        reward = obj_value - self._prev_obj_value
        self._prev_obj_value = obj_value

        if self.objective.minimize:
            reward = -reward

        truncated = False
        terminal = state.is_terminal()
        info = self.get_info()

        return self._observation, reward, terminal, truncated, info

    def render(self) -> None:
        if self._status != RUNNING:
            raise RuntimeError(
                "Cannot render an environment during configuration."
            )

        self.renderer.render(self.state)

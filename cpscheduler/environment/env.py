"""
env.py

This module defines the SchedulingEnv class, which is a custom environment for generic
scheduling problems.
It is designed to be modular and extensible, allowing users to define their own scheduling
problems by specifying the machine setup, constraints, objectives, and instances.

The environment is based on the OpenAI Gym interface, making it compatible with various
reinforcement learning libraries. It provides methods for resetting the environment, taking
steps, rendering the environment, and exporting the scheduling model.
"""

from typing import Any
from collections.abc import Iterable, Mapping
from typing_extensions import TypeIs

from cpscheduler.utils.list_utils import convert_to_list
from cpscheduler.utils._protocols import Metric

from cpscheduler.environment._common import (
    InstanceTypes,
    TIME,
    InfoType,
    ObsType,
    Options,
    MAX_TIME,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.instructions import (
    parse_instruction,
    ActionType,
    is_single_action,
    Schedule,
    QueueControl
)
from cpscheduler.environment.schedule_setup import ScheduleSetup
from cpscheduler.environment.constraints import Constraint, PassiveConstraint
from cpscheduler.environment.objectives import Objective

from cpscheduler.environment._render import Renderer


def prepare_instance(instance: InstanceTypes) -> dict[str, list[Any]]:
    "Prepare the instance data to a standard dictionary format."
    return {str(feature): convert_to_list(instance[feature]) for feature in instance}


def is_info_dict(value: Any) -> TypeIs[Mapping[str, Any]]:
    "Type guard to check if a value is an info dictionary."
    return isinstance(value, Mapping) and all(
        isinstance(k, str) for k in value.keys()  # pyright: ignore[reportUnknownVariableType]
    )

class SchedulingEnv:
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
    setup_constraints: list[Constraint]
    passive_constraints: list[PassiveConstraint]
    objective: Objective

    metrics: dict[str, Metric[object | Mapping[str, Any]]]

    renderer: Renderer

    # Environment dynamic variables
    state: ScheduleState
    schedule: Schedule

    # Helper variables
    _prev_obj_value: float
    event_counter: int
    force_reset: bool

    def __init__(
        self,
        machine_setup: ScheduleSetup,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        instance: InstanceTypes | None = None,
        metrics: Mapping[str, Metric[Any]] | None = None,
        render_mode: Renderer | str | None = None,
    ):
        self.setup = machine_setup

        self.state = ScheduleState()
        self.schedule = Schedule()

        self.constraints = []
        self.setup_constraints = []
        self.passive_constraints = []
        if constraints is not None:
            for constraint in constraints:
                self.add_constraint(constraint)

        if objective is None:
            objective = Objective()

        self.metrics = {}
        if metrics is not None:
            for name, metric in metrics.items():
                self.add_metric(name, metric)

        self.set_objective(objective)

        if instance is not None:
            self.set_instance(instance)

        self.renderer = (
            render_mode if isinstance(render_mode, Renderer) else Renderer.get_renderer(render_mode)
        )

        self._prev_obj_value = 0.0
        self.event_counter = 0
        self.force_reset = True

    def __repr__(self) -> str:
        if self.state.loaded:
            return (
                f"SchedulingEnv({self.get_entry()}, n_tasks={self.state.n_tasks}, "
                f"current_time={self.current_time}, objective={self.get_objective()})"
            )

        return f"SchedulingEnv({self.get_entry()}, n_tasks=0)"

    @property
    def current_time(self) -> TIME:
        "Get the current time in the environment."
        return self.state.time

    # Environment configuration public methods
    def add_constraint(self, constraint: Constraint) -> None:
        "Add a constraint to the environment."
        if self.state.loaded:
            constraint.initialize(self.state)

        if isinstance(constraint, PassiveConstraint):
            self.passive_constraints.append(constraint)

        else:
            self.constraints.append(constraint)

        self.force_reset = True

    def set_objective(self, objective: Objective) -> None:
        "Set the objective function for the environment."
        if self.state.loaded:
            objective.initialize(self.state)

        self.objective = objective

    def set_instance(self, instance: InstanceTypes) -> None:
        """
        Set the instance data for the environment.

        Parameters:
            instance: InstanceTypes
                The instance data for the scheduling problem, can be a DataFrame or a dictionary.
        """
        instance_dict = prepare_instance(instance)

        self.state.read_instance(instance_dict)
        self.setup.initialize(self.state)

        self.setup_constraints.clear()
        for constraint in self.setup.setup_constraints(self.state):
            constraint.initialize(self.state)
            self.setup_constraints.append(constraint)

        for p_constraint in self.passive_constraints:
            p_constraint.initialize(self.state)

        for constraint in self.constraints:
            constraint.initialize(self.state)

        self.objective.initialize(self.state)

        self.force_reset = True

    def add_metric(self, name: str, metric: Metric[Any]) -> None:
        "Add a metric to the environment."
        self.metrics[name] = metric

    def clear_instance(self) -> None:
        "Clear the current instance data from the environment."
        self.state.clear()
        self.force_reset = True

    def clear_constraints(self) -> None:
        "Clear all constraints from the environment."
        self.constraints.clear()
        self.setup_constraints.clear()
        self.passive_constraints.clear()
        self.force_reset = True

    def clear_metrics(self) -> None:
        "Clear all metrics from the environment."
        self.metrics.clear()

    def clear(self) -> None:
        "Clear all instance data, constraints, objectives, and metrics from the environment."
        self.clear_instance()
        self.clear_constraints()
        self.clear_metrics()

        self.set_objective(Objective())

    def get_entry(self) -> str:
        "Get a string representation of the environment's configuration."
        alpha = self.setup.get_entry()

        beta = ",".join([constraint.get_entry() for constraint in self.constraints])

        gamma = self.objective.get_entry()

        return f"{alpha}|{beta}|{gamma}"

    # Environment state retrieval methods
    def get_state(self) -> ObsType:
        "Retrieve the current state of the environment from tasks."
        return self.state.get_observation()

    def get_info(self) -> InfoType:
        "Retrieve additional information about the environment."
        info: dict[str, Any] = {
            "current_time": self.current_time,
            "objective_value": self._prev_obj_value,
            "event_count": self.event_counter,
        }

        for metric_name, metric in self.metrics.items():
            metric_value = metric(self.state)

            if is_info_dict(metric_value):
                info.update({f"{metric_name}_{key}": value for key, value in metric_value.items()})

            else:
                info[metric_name] = metric_value

        return info

    # Environment internal methods for simulation
    def schedule_action(self, action: ActionType) -> None:
        "Consume an action by adding the corresponding instruction(s) to the schedule."
        if action is None:
            return

        if is_single_action(action):
            action = [action]

        for single_action, *args in action:
            instruction, time = parse_instruction(single_action, args, self.state)

            if 0 <= time < self.current_time:
                raise ValueError(
                    f"Cannot schedule instruction {instruction} at past time {time} "
                    f"from current time {self.current_time}."
                )

            self.schedule.add_instruction(instruction, time)

    def propagate(self) -> None:
        "Propagate the new bounds through the constraints until a fixed-point is reached."
        while self.state.event_queue:
            self.event_counter += 1
            event = self.state.event_queue.popleft()

            for constraint in self.setup_constraints:
                constraint.propagate(event, self.state)

            for constraint in self.constraints:
                constraint.propagate(event, self.state)

    def advance_clock(self) -> None:
        next_time = MAX_TIME

        next_instruction_time = self.schedule.get_next_instruction_time()
        if next_instruction_time < next_time:
            next_time = next_instruction_time

        if self.state.awaiting_tasks:
            next_decision_time = self.state.get_next_available_time(strict=True)

            if next_decision_time < next_time:
                next_time = next_decision_time

        if next_time == MAX_TIME and self.state.fixed_tasks:
            next_time = self.state.get_next_completion_time()

        if next_time > self.current_time:
            self.state.time = next_time

    # Environment API methods
    def reset(self, *, options: Options = None) -> tuple[ObsType, InfoType]:
        if isinstance(options, dict) and "instance" in options:
            self.set_instance(options["instance"])

        if not self.state.loaded:
            raise ValueError(
                "Environment has not been loaded with an instance. "
                "Please call reset(options={'instance':<instance>}) or set_instance(<instance>) before resetting."
            )

        self.schedule.reset()
        self.state.reset()

        for constraint in self.constraints:
            constraint.reset(self.state)

        for constraint in self.setup_constraints:
            constraint.reset(self.state)

        self._prev_obj_value = self.get_objective()
        self.event_counter = 0
        self.force_reset = False

        self.propagate()

        return self.get_state(), self.get_info()

    def step(self, action: ActionType = None) -> tuple[ObsType, float, bool, bool, InfoType]:
        if not self.state.loaded or self.force_reset:
            raise RuntimeError(
                "Environment was not reset after loading an instance, or wasn't loaded. "
                "Please call reset(options={'instance':<instance>}) or set_instance(<instance>), then reset()."
            )

        self.schedule_action(action)

        while not self.schedule.is_empty():
            # Invariant: Each iteration has the time static during instruction processing

            control: QueueControl = QueueControl.CONTINUE
            for instruction_result in self.schedule.instruction_queue(self.state):
                # After each instruction is processed, ensure domains are updated until a fixed point
                # is reached before processing the next instruction.
                control = instruction_result.queue_control

                # if instruction_result.done:
                if self.state.event_queue:
                    self.propagate()

            if control == QueueControl.INTERRUPT:
                # If the schedule processing was interrupted due to a instruction, do not advance
                # the time and allow the agent to react to the new state. 
                break

            if self.schedule.is_empty():
                if self.state.get_next_available_time() <= self.current_time:
                    break

            self.advance_clock()

        # Gymnasium-like step return
        obs = self.get_state()

        obj_value = self.get_objective()
        reward = obj_value - self._prev_obj_value
        self._prev_obj_value = obj_value

        if self.objective.minimize:
            reward = -reward

        truncated = False
        terminal = self.state.is_terminal()
        info = self.get_info()

        return obs, reward, terminal, truncated, info

    def render(self) -> None:
        self.renderer.render(self.current_time, self.state)

    def get_objective(self) -> float:
        "Get the current value of the objective function."
        return self.objective.get_current(self.state)

    # Custom serialization methods for pickling and deep copying
    def __reduce__(self) -> Any:
        """
        Custom reduce method to ensure the environment can be pickled and deep copied correctly.
        This is necessary for compatibility with multiprocessing and other serialization
        mechanisms.
        """
        return (
            self.__class__,
            (
                self.setup,
                None,
                None,
                None,  # instance_config will be set later
                self.metrics,
                self.renderer,
            ),
            (
                self.constraints,
                self.setup_constraints,
                self.passive_constraints,
                self.objective,
                self.metrics,
                self.renderer,
                self.state,
                self.schedule,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        """
        Custom setstate method to restore the environment's state after unpickling.
        This is necessary to ensure the environment is correctly initialized with its
        data and tasks.
        """
        (
            self.constraints,
            self.setup_constraints,
            self.passive_constraints,
            self.objective,
            self.metrics,
            self.renderer,
            self.state,
            self.schedule,
        ) = state

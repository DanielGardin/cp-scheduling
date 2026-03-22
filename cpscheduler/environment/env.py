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
from cpscheduler.utils._protocols import (
    Metric,
    InstanceTypes,
    InfoType,
    Options,
)

from cpscheduler.environment.state import ScheduleState, ObsType
from cpscheduler.environment.state.events import VarField
from cpscheduler.environment.des import (
    ActionType,
    Schedule,
    parse_instruction,
    is_single_action,
)
from cpscheduler.environment.schedule_setup import ScheduleSetup
from cpscheduler.environment.constraints import Constraint, PassiveConstraint
from cpscheduler.environment.objectives import Objective

from cpscheduler.environment._render import Renderer


def prepare_instance(instance: InstanceTypes) -> dict[str, list[Any]]:
    "Prepare the instance data to a standard dictionary format."
    return {
        str(feature): convert_to_list(instance[feature]) for feature in instance
    }


def is_info_dict(value: Any) -> TypeIs[Mapping[Any, Any]]:
    "Type guard to check if a value is an info dictionary."
    return isinstance(value, Mapping)

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
    combined_constraints: list[Constraint]
    objective: Objective

    metrics: dict[str, Metric[object | Mapping[str, Any]]]

    renderer: Renderer

    # Environment dynamic variables
    state: ScheduleState
    schedule: Schedule

    # Helper variables
    _prev_obj_value: float
    event_count: int
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
        self.combined_constraints = []
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
            render_mode
            if isinstance(render_mode, Renderer)
            else Renderer.get_renderer(render_mode)
        )

        self._prev_obj_value = 0.0
        self.event_count = 0
        self.force_reset = True

    def __repr__(self) -> str:
        state = self.state

        if state.loaded:
            return (
                f"SchedulingEnv({self.get_entry()}, n_tasks={state.n_tasks}, "
                f"current_time={state.time}, objective={self.get_objective()})"
            )

        return f"SchedulingEnv({self.get_entry()}, n_tasks=0)"

    # Environment configuration public methods
    def add_constraint(self, constraint: Constraint) -> None:
        "Add a constraint to the environment."
        state = self.state

        if state.loaded:
            constraint.initialize(state)

        if isinstance(constraint, PassiveConstraint):
            self.passive_constraints.append(constraint)

        else:
            self.constraints.append(constraint)
            self._rebuild_combined_constraints()

        self.force_reset = True

    def set_objective(self, objective: Objective) -> None:
        "Set the objective function for the environment."
        state = self.state

        if state.loaded:
            objective.initialize(state)

        self.objective = objective

    def set_instance(self, instance: InstanceTypes) -> None:
        """
        Set the instance data for the environment.

        Parameters:
            instance: InstanceTypes
                The instance data for the scheduling problem, can be a DataFrame or a dictionary.
        """
        instance_dict = prepare_instance(instance)

        state = self.state
        setup = self.setup

        state.read_instance(instance_dict)
        setup.initialize(state)

        self.setup_constraints.clear()
        for constraint in setup.setup_constraints(state):
            self.setup_constraints.append(constraint)

        self._rebuild_combined_constraints()

        for p_constraint in self.passive_constraints:
            p_constraint.initialize(state)

        for constraint in self.combined_constraints:
            constraint.initialize(state)

        self.objective.initialize(state)

        self.force_reset = True

    def add_metric(self, name: str, metric: Metric[Any]) -> None:
        "Add a metric to the environment."
        self.metrics[name] = metric

    def _rebuild_combined_constraints(self) -> None:
        "Rebuild the combined constraint list for the propagation loop."
        self.combined_constraints = self.setup_constraints + self.constraints

    def clear_constraints(self) -> None:
        "Clear all constraints from the environment."
        self.constraints.clear()
        self.setup_constraints.clear()
        self.passive_constraints.clear()
        self.combined_constraints.clear()
        self.force_reset = True

    def clear_metrics(self) -> None:
        "Clear all metrics from the environment."
        self.metrics.clear()

    def clear(self) -> None:
        "Clear all instance data, constraints, objectives, and metrics from the environment."
        self.clear_constraints()
        self.clear_metrics()

        self.set_objective(Objective())

    def get_entry(self) -> str:
        "Get a string representation of the environment's configuration."
        alpha = self.setup.get_entry()

        beta = ",".join(
            [constraint.get_entry() for constraint in self.constraints]
            + [
                constraint.get_entry()
                for constraint in self.passive_constraints
            ]
        )

        gamma = self.objective.get_entry()

        return f"{alpha}|{beta}|{gamma}"

    # Environment state retrieval methods
    def get_state(self) -> ObsType:
        "Retrieve the current state of the environment from tasks."
        return self.state.get_observation()

    def get_info(self) -> InfoType:
        "Retrieve additional information about the environment."
        info: dict[str, Any] = {
            "current_time": self.state.time,
            "objective_value": self._prev_obj_value,
            "event_count": self.event_count,
        }

        for metric_name, metric in self.metrics.items():
            metric_value = metric(self.state)

            if is_info_dict(metric_value):
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

        if is_single_action(action):
            action = [action]

        state = self.state

        for intruction_args in action:
            instruction, time, priority = parse_instruction(intruction_args)
            self.schedule.add_event(instruction, state, time, priority)

    def propagate(self) -> None:
        "Propagate the new bounds through the constraints until a fixed-point is reached."
        state = self.state
        event_queue = state.event_queue
        combined = self.combined_constraints

        idx = 0
        while idx < len(event_queue):
            event = event_queue[idx]
            task_id = event.task_id
            machine_id = event.machine_id

            match event.field:
                case VarField.ASSIGNMENT:
                    for constraint in combined:
                        constraint.on_assignment(task_id, machine_id, state)

                case VarField.START_LB:
                    for constraint in combined:
                        constraint.on_start_lb(task_id, machine_id, state)

                case VarField.START_UB:
                    for constraint in combined:
                        constraint.on_start_ub(task_id, machine_id, state)

                case VarField.END_LB:
                    for constraint in combined:
                        constraint.on_end_lb(task_id, machine_id, state)

                case VarField.END_UB:
                    for constraint in combined:
                        constraint.on_end_ub(task_id, machine_id, state)

                case VarField.PRESENCE:
                    for constraint in combined:
                        constraint.on_presence(task_id, state)

                case VarField.ABSENCE:
                    for constraint in combined:
                        constraint.on_absence(task_id, state)

                case VarField.INFEASIBLE:
                    for constraint in combined:
                        constraint.on_infeasibility(task_id, machine_id, state)

                case _:
                    raise ValueError(f"Unknown event field: {event.field}")

            idx += 1

        self.event_count += idx
        state.event_queue.clear()


    def advance_clock(self) -> bool:
        schedule = self.schedule
        state = self.state

        empty_schedule = schedule.is_empty()

        if not empty_schedule:
            next_time = schedule.next_time()

        else:
            if state.runtime_state.awaiting_tasks:
                next_time = state.get_next_start_lb()

            else:
                next_time = state.get_last_completion_time()


        self.state.advance_time(next_time)

        # TODO: The only way a infeasible action can currently be detected
        # is if it causes the schedule to indefinitely postpone events
        # Locked semantics can help, we don't need to worry about
        # tighetning bounds at each time step because no events can be
        # wrongly processed at an intermediate time.
        # for task_id in state.runtime_state.awaiting_tasks:
        #     start_lb = state.get_start_lb(task_id)

        #     if start_lb <= state.time:
        #         state.tight_start_lb(task_id, state.time)

        for constraint in self.combined_constraints:
            constraint.on_time_update(next_time, self.state)

        self.propagate()

        return not empty_schedule

    # Environment API methods
    def reset(self, *, options: Options = None) -> tuple[ObsType, InfoType]:
        if isinstance(options, dict) and "instance" in options:
            self.set_instance(options["instance"])

        if not self.state.loaded:
            raise ValueError(
                "Environment has not been loaded with an instance. "
                "Please call reset(options={'instance':<instance>}) or set_instance(<instance>) before resetting."
            )

        state = self.state

        self.schedule.reset()
        state.reset()

        for constraint in self.combined_constraints:
            constraint.reset(state)

        self._prev_obj_value = self.get_objective()
        self.event_count = 0
        self.force_reset = False

        self.propagate()

        return self.get_state(), self.get_info()

    def step(
        self, action: ActionType = None
    ) -> tuple[ObsType, float, bool, bool, InfoType]:
        state = self.state

        if not state.loaded or self.force_reset:
            raise RuntimeError(
                "Environment was not reset after loading an instance, or wasn't loaded. "
                "Please call reset(options={'instance':<instance>}) or set_instance(<instance>), then reset()."
            )

        self.schedule_action(action)

        schedule = self.schedule

        while self.advance_clock() and not state.is_terminal():
            # Invariant: Each iteration has the time static during instruction processing
            for event in schedule.instruction_queue(state):
                # After each instruction is processed, ensure domains are updated until a fixed point
                # is reached before processing the next instruction.
                # control = instruction_result.queue_control
                event.process(state, schedule)

                self.propagate()

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
        self.renderer.render(self.state)

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
                self.combined_constraints,
                self.objective,
                self.metrics,
                self.renderer,
                self.state,
                self.schedule,
                self._prev_obj_value,
                self.event_count,
                self.force_reset,
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
            self.combined_constraints,
            self.objective,
            self.metrics,
            self.renderer,
            self.state,
            self.schedule,
            self._prev_obj_value,
            self.event_count,
            self.force_reset,
        ) = state

"""
env.py

This module defines the SchedulingEnv class, which is a custom environment for generic
cheduling problems.
It is designed to be modular and extensible, allowing users to define their own scheduling
problems by specifying the machine setup, constraints, objectives, and instances.

The environment is based on the OpenAI Gym interface, making it compatible with various
reinforcement learning libraries. It provides methods for resetting the environment, taking
steps, rendering the environment, and exporting the scheduling model.
"""

from warnings import warn

from typing import Any
from collections.abc import Iterable, Mapping

from mypy_extensions import u8, i64

from cpscheduler.utils.list_utils import convert_to_list
from cpscheduler.utils._protocols import Metric

from cpscheduler.environment._common import (
    MAX_INT,
    InstanceTypes,
    TASK_ID,
    TIME,
    InfoType,
    ObsType,
    Options,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.instructions import (
    Instruction,
    Signal,
    parse_instruction,
    Action,
    ActionType,
    is_single_action,
)
from cpscheduler.environment.schedule_setup import ScheduleSetup, setups
from cpscheduler.environment.constraints import Constraint, constraints
from cpscheduler.environment.objectives import Objective, objectives

from cpscheduler.environment._render import Renderer


def prepare_instance(instance: InstanceTypes) -> dict[str, list[Any]]:
    "Prepare the instance data to a standard dictionary format."
    return {str(feature): convert_to_list(instance[feature]) for feature in instance}


class SchedulingEnv:
    """
    SchedulingEnv is a custom environment for generic scheduling problems. It is designed to be
    modular and extensible, allowing users to define their own scheduling problems by specifying
    the machine setup, constraints, objectives, and instances.

    The environment is based on the OpenAI Gym interface, making it compatible with various
    reinforcement learning libraries. It provides methods for resetting the environment, taking
    steps, rendering the environment, and exporting the scheduling model.

    Attributes:
        machine_setup: ScheduleSetup
            The machine setup for the scheduling problem.

        constraints: Iterable of Constraint
            A list of constraints to be applied to the scheduling problem.

        objective: Objective
            The objective function to be optimized during scheduling.

        render_mode: str
            The rendering mode for visualizing the scheduling process.

        minimize: bool, optional
            Whether to minimize or maximize the objective function. Default depends on the chosen
            objective.

        allow_preemption: bool, optional, default=False
            Whether to allow preemption in the scheduling process. If True, tasks can be
            interrupted and resumed later.

        instance: InstanceTypes, optional
            The instance data for the scheduling problem. It can be a DataFrame or a dictionary
            containing task features and their values.

        processing_times: ProcessTimeAllowedTypes, optional
            The processing times for the tasks, it is dependent on the machine setup.
            If not provided, the environment will attempt to infer processing times from the
            instance data.

        job_instance: InstanceTypes, optional
            The job instance data for the scheduling problem. It can be a DataFrame or a dictionary
            containing job features and their values. If None, no job instance is set.

        job_ids: Iterable[int] | str, optional
            The job IDs for the tasks. If None, job IDs are as the default index of job_instance.

        n_parts: int, optional
            The number of parts to split the tasks into. If None, it defaults to 16 if preemption is
            allowed, otherwise 1.
    """

    # Environment static variables
    setup: ScheduleSetup
    constraints: dict[str, Constraint]
    objective: Objective

    metrics: dict[str, Metric[Any]]

    renderer: Renderer

    # Environment dynamic variables
    state: ScheduleState
    schedule: dict[TASK_ID, list[Instruction]]
    current_time: TIME
    advancing_to: TIME
    query_times: list[TIME]
    force_reset: bool = False

    def __init__(
        self,
        machine_setup: ScheduleSetup,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        instance: InstanceTypes | None = None,
        metrics: Mapping[str, Metric[Any]] | None = None,
        render_mode: Renderer | str | None = None,
        allow_preemption: bool = False,
    ):
        self.state = ScheduleState()
        self.state.set_preemptive(allow_preemption)
        self.preemptive = allow_preemption
        self.setup = machine_setup

        self.constraints = {}
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

        self.schedule = {-1: []}

        self.current_time = 0
        self.advancing_to = 0
        self.query_times: list[TIME] = []

        self.renderer = (
            render_mode
            if isinstance(render_mode, Renderer)
            else Renderer.get_renderer(render_mode)
        )

        if instance is not None:
            self.set_instance(instance)

    def __repr__(self) -> str:
        if self.state.loaded:
            return (
                f"SchedulingEnv({self.get_entry()}, n_tasks={self.state.n_tasks}, "
                f"current_time={self.current_time}, objective={self._get_objective()})"
            )

        return f"SchedulingEnv({self.get_entry()}, n_tasks=0)"

    # Environment configuration public methods
    def add_constraint(self, constraint: Constraint, replace: bool = False) -> None:
        "Add a constraint to the environment."
        name = constraint.name

        if name in self.constraints and not replace:
            raise ValueError(
                f"Constraint with name {name} already exists. Please use a different name."
            )

        if self.state.loaded:
            constraint.initialize(self.state)

        self.constraints[name] = constraint

    def set_objective(self, objective: Objective) -> None:
        "Set the objective function for the environment."
        if self.state.loaded:
            objective.initialize(self.state)

        self.objective = objective

    def add_metric(self, name: str, metric: Metric[Any]) -> None:
        "Add a metric to the environment."
        self.metrics[name] = metric

    def set_instance(self, instance: InstanceTypes) -> None:
        """
        Set the instance data for the environment.

        Parameters:
            instance: InstanceTypes
                The instance data for the scheduling problem, can be a DataFrame or a dictionary.
        """
        instance = prepare_instance(instance)

        self.state.read_instance(instance)
        self.setup.initialize(self.state)

        self.state.set_n_machines(self.setup.n_machines)

        for constraint in self.setup.setup_constraints(self.state):
            self.add_constraint(constraint, replace=True)

        self.objective.initialize(self.state)
        for constraint in self.constraints.values():
            constraint.initialize(self.state)

        self.force_reset = True

    def get_entry(self) -> str:
        "Get a string representation of the environment's configuration."
        alpha = self.setup.get_entry()

        beta = ",".join(
            [
                constraint.get_entry()
                for constraint in self.constraints.values()
                if not constraint.name.startswith("__") and constraint.get_entry()
            ]
        )

        if self.preemptive:
            beta += f"{',' if beta else ''}prmp"

        gamma = self.objective.get_entry()

        return f"{alpha}|{beta}|{gamma}"

    # Environment state retrieval methods
    def _get_state(self) -> ObsType:
        "Retrieve the current state of the environment from tasks."
        return self.state.get_observation(self.current_time)

    def _get_info(self) -> InfoType:
        "Retrieve additional information about the environment."
        objective_value = self._get_objective()

        info = {
            "n_queries": len(self.query_times),
            "current_time": int(self.current_time),
        }

        for metric_name, metric in self.metrics.items():
            metric_value = metric(self.current_time, self.state, objective_value)

            if isinstance(metric_value, Mapping):
                info.update(metric_value)

            else:
                info[metric_name] = metric_value

        return info

    def _propagate(self) -> None:
        "Propagate the new bounds through the constraints"
        # Currently not sure if this is needed, but keeping it for future reference.

        # for task_id in self.state.awaiting_tasks:
        #     task = self.state[task_id]
        #     if task.get_start_lb() < self.current_time:
        #         task.set_start_lb(self.current_time)

        for constraint in self.constraints.values():
            constraint.propagate(self.current_time, self.state)

        self.state.transition_tasks.clear()

    # Environment API methods
    def reset(self, *, options: Options = None) -> tuple[ObsType, InfoType]:
        if isinstance(options, dict) and "instance" in options:
            self.set_instance(options["instance"])

        if not self.state.loaded:
            raise ValueError(
                "Environment has not been loaded with an instance. "
                "Please call reset(options={'instance':<instance>}) or set_instance(<instance>) before resetting."
            )

        self.schedule.clear()
        self.schedule[-1] = []

        self.current_time = 0
        self.advancing_to = 0
        self.query_times.clear()

        self.state.reset()
        for constraint in self.constraints.values():
            constraint.reset(self.state)

        self._propagate()

        self.force_reset = False

        return self._get_state(), self._get_info()

    def step(
        self,
        action: ActionType = None,
    ) -> tuple[ObsType, float, bool, bool, InfoType]:
        if not self.state.loaded or self.force_reset:
            raise RuntimeError(
                "Environment was not reset after loading an instance, or wasn't loaded. "
                "Please call reset(options={'instance':<instance>}) or set_instance(<instance>), then reset()."
            )

        if is_single_action(action):
            single_args = tuple(map(int, action[1:]))
            self._schedule_instruction(action[0], single_args)

        elif action is not None:
            for instruction in action:
                args = tuple(map(int, instruction[1:]))
                self._schedule_instruction(instruction[0], args)

        self.query_times.append(self.current_time)
        previous_objective = self._get_objective()

        while True:
            # In dynamic usage, this runs twice as necessary (checking for an empty schedule)
            # Ideally, we want to break this loop whenever the schedule is empty or an instruction halts.
            if self._dispatch_instruction():
                break

            self.render()

        # Heavy: 30% of step time in dynamic usage.
        obs = self._get_state()

        reward = self._get_objective() - previous_objective
        if self.objective.minimize:
            reward = -reward

        truncated = False
        terminal = self._is_terminal()
        info = self._get_info()

        return obs, reward, terminal, truncated, info

    def render(self) -> None:
        self.renderer.render(self.current_time, self.state)

    def _get_objective(self) -> float:
        "Get the current value of the objective function."
        return float(self.objective.get_current(self.current_time, self.state))

    def _is_terminal(self) -> bool:
        "Check if the environment is in a terminal state."
        if self.state.awaiting_tasks:
            return False

        return all(
            task.is_completed(self.current_time) for task in self.state.fixed_tasks
        )

    def _is_schedule_empty(self) -> bool:
        return not self.schedule[-1] and len(self.schedule) == 1

    def _advance_to_decision_point(self, strict: bool = False) -> None:
        "Obtain the next decision time to advance. If strict, only consider future tasks."
        next_time = (
            MAX_INT if self.current_time >= self.advancing_to else self.advancing_to
        )

        if strict:
            for task in self.state.awaiting_tasks:
                start_lb = task.get_start_lb()

                if self.current_time < start_lb < next_time:
                    next_time = start_lb

            for instruction_time in self.schedule:
                if self.current_time < instruction_time < next_time:
                    next_time = instruction_time

        else:
            for task in self.state.awaiting_tasks:
                start_lb = task.get_start_lb()

                if start_lb <= self.current_time:
                    return

                if start_lb < next_time:
                    next_time = start_lb

            for instruction_time in self.schedule:
                if self.current_time <= instruction_time < next_time:
                    next_time = instruction_time

        if next_time == MAX_INT:
            self.current_time = max(
                task.get_end_ub() for task in self.state.fixed_tasks
            )

        elif next_time > self.current_time:
            self.current_time = next_time

    def _advance_to_next_instruction(self) -> None:
        "Advance the environment to the next instruction time."
        next_time = self.advancing_to
        for instruction_time in self.schedule:
            if self.current_time <= instruction_time < next_time:
                next_time = instruction_time

        self.current_time = next_time

    def _schedule_instruction(
        self, action: str | Instruction, args: tuple[int, ...]
    ) -> None:
        "Add a single instruction to the schedule."
        instruction, time = parse_instruction(action, args, self.state)

        if 0 <= time < self.current_time:
            warn(
                f"Scheduled instruction {instruction} with arguments {args} is in the past. \
                It will be executed immediately."
            )
            time = self.current_time

        if time not in self.schedule:
            self.schedule[time] = []

        self.schedule[time].append(instruction)

    def _process_next_instruction(
        self, scheduled_time: TIME = -1, allow_wait: bool = True
    ) -> bool:
        """
        Process the next instruction in the schedule at the given time.
        If time is -1, use the current time and if allow_wait is False,
        it halts the environment when an instruction is requested to be waited.
        """
        instruction = Instruction()
        signal = Signal(Action.SKIPPED)

        i: i64 = -1
        schedule = self.schedule[scheduled_time]

        while signal.action & Action.SKIPPED and i + 1 < len(schedule):
            i += 1
            instruction = schedule[i]

            signal = instruction.process(self.current_time, self.state, self.schedule)

        action: u8 = signal.action
        if not allow_wait and action == Action.WAIT:
            warn(f"{instruction} is not allowed to wait at {self.current_time}.")
            schedule.pop(i)
            return True

        if action & Action.SKIPPED:
            if any(
                task.is_executing(self.current_time) for task in self.state.fixed_tasks
            ):
                action = Action.WAIT

            else:
                action = Action.HALT

        if not (action & Action.SKIPPED) and i != -1:
            schedule.pop(i)

        if action & Action.REEVALUATE:
            for task in self.state.awaiting_tasks:
                task.set_start_lb(self.current_time)

        if action & Action.PROPAGATE:
            self._propagate()

        if action & Action.ADVANCE:
            if signal.time > 0:
                self.advancing_to = signal.time
                self._advance_to_next_instruction()

            else:
                self._advance_to_decision_point(strict=False)

        if action & Action.ADVANCE_NEXT:
            self._advance_to_decision_point(strict=True)

        if action & Action.RAISE:
            warn(
                f"Error in instruction {instruction} at time {self.current_time}: {signal.info}"
            )

        return bool(action & Action.HALT)

    def _dispatch_instruction(self) -> bool:
        "Dispatch the next instruction in the schedule depending on the current state."
        if self.current_time in self.schedule:
            halt = self._process_next_instruction(self.current_time, False)

            if not self.schedule[self.current_time]:
                self.schedule.pop(self.current_time)

            return halt

        if self.current_time < self.advancing_to:
            self._advance_to_next_instruction()
            return False

        if self._is_schedule_empty():
            if not self.state.awaiting_tasks:
                for task in self.state.fixed_tasks:
                    end_time = task.get_end_ub()
                    if end_time > self.current_time:
                        self.current_time = end_time

            return True

        return self._process_next_instruction()

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
                self.preemptive,
            ),
            (
                self.constraints,
                self.objective,
                self.metrics,
                self.renderer,
                self.state,
                self.schedule,
                self.current_time,
                self.advancing_to,
                self.query_times,
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
            self.objective,
            self.metrics,
            self.renderer,
            self.state,
            self.schedule,
            self.current_time,
            self.advancing_to,
            self.query_times,
        ) = state

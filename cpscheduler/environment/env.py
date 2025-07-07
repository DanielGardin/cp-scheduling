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
from typing_extensions import TypeIs

from gymnasium import Env
from gymnasium.spaces import Tuple, Text, Box, OneOf

from mypy_extensions import u8, i64

from ._common import (
    MAX_INT,
    ProcessTimeAllowedTypes,
    TASK_ID,
    PART_ID,
    TIME,
    InstanceTypes,
    InfoType,
    ObsType,
)
from .data import SchedulingData
from .tasks import Tasks
from .instructions import (
    Instruction,
    Signal,
    parse_instruction,
    Action,
    ActionType,
    SingleAction,
)
from .schedule_setup import ScheduleSetup
from .constraints import Constraint
from .objectives import Objective
from .utils import convert_to_list, is_iterable_int

from ._render import Renderer, PlotlyRenderer

# Define the action space for the environment
InstructionSpace = Text(max_length=10)
IntSpace = Box(low=0, high=int(MAX_INT), shape=(), dtype=int)  # type: ignore
# do not want numpy dependency here

ActionSpace = OneOf(
    [
        Tuple([InstructionSpace]),
        Tuple([InstructionSpace, IntSpace]),
        Tuple([InstructionSpace, IntSpace, IntSpace]),
        Tuple([InstructionSpace, IntSpace, IntSpace, IntSpace]),
    ]
)


def is_single_action(
    action: ActionType,
) -> TypeIs[SingleAction]:
    "Check if the action is a single instruction or a iterable of instructions."
    if not isinstance(action, tuple):
        return False

    if action:
        return isinstance(action[0], str) and is_iterable_int(action[1:])

    return True


def prepare_instance(instance: InstanceTypes) -> dict[str, list[Any]]:
    "Prepare the instance data to a standard dictionary format."
    features = instance.keys() if isinstance(instance, Mapping) else instance.columns

    return {feature: convert_to_list(instance[feature]) for feature in features}


class SchedulingEnv(Env[ObsType, ActionType]):
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
            Whether to allow preemption in the scheduling process. If True, tasks can be interrupted
            and resumed later.

        instance: InstanceTypes, optional
            The instance data for the scheduling problem. It can be a DataFrame or a dictionary
            containing task features and their values.

        processing_times: ProcessTimeAllowedTypes, optional
            The processing times for the tasks, it is dependent on the machine setup. If not provided,
            the environment will attempt to infer processing times from the instance data.

        job_instance: InstanceTypes, optional
            The job instance data for the scheduling problem. It can be a DataFrame or a dictionary
            containing job features and their values. If None, no job instance is set.

        job_ids: Iterable[int] | str, optional
            The job IDs for the tasks. If None, job IDs are as the default index of the job_instance.

        n_parts: int, optional
            The number of parts to split the tasks into. If None, it defaults to 16 if preemption is
            allowed, otherwise 1.
    """

    # Environment static variables
    data: SchedulingData
    setup: ScheduleSetup
    constraints: dict[str, Constraint]
    objective: Objective
    allow_preemption: bool
    minimize: bool
    loaded: bool

    # Environment dynamic variables
    tasks: Tasks
    schedule: dict[TASK_ID, list[Instruction]]
    current_time: TIME
    advancing_to: TIME
    query_times: list[TIME]

    # Gymnasium support variables
    renderer: Renderer
    _metadata: dict[str, Any]

    # Environment constructor methods
    def __init__(
        self,
        machine_setup: ScheduleSetup,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        render_mode: Renderer | str | None = None,
        *,
        minimize: bool | None = None,
        allow_preemption: bool = False,
        instance: InstanceTypes | None = None,
        processing_times: ProcessTimeAllowedTypes = None,
        job_instance: InstanceTypes | None = None,
        job_feature: str = "",
        n_parts: int | None = None,
    ):
        self.allow_preemption = allow_preemption
        self.loaded = False

        self.setup = machine_setup

        self.constraints = {}
        if constraints is not None:
            for constraint in constraints:
                self.add_constraint(constraint)

        self.set_objective(
            Objective() if objective is None else objective, minimize=minimize
        )

        self.schedule = {-1: []}

        self.current_time = 0
        self.query_times = []

        self._metadata = {
            "render_modes": ["human"],
            "render_fps": 50,
        }

        self.renderer = (
            render_mode
            if isinstance(render_mode, Renderer)
            else self._dispatch_render(render_mode)
        )

        self.action_space = ActionSpace

        self.advancing_to = 0

        if instance is not None:
            self.set_instance(
                instance, processing_times, job_instance, job_feature, n_parts
            )

    def _dispatch_render(self, render_model: str | None) -> Renderer:
        "Dispatch the renderer based on the render model."
        if render_model is None:
            return Renderer()

        if render_model == "human":
            return PlotlyRenderer()

        raise ValueError(f"Unknown render model: {render_model}. Supported: 'human'.")

    # Environment configuration public methods
    def add_constraint(self, constraint: Constraint, replace: bool = False) -> None:
        "Add a constraint to the environment."
        name = constraint.name

        if name in self.constraints and not replace:
            raise ValueError(
                f"Constraint with name {name} already exists. Please use a different name."
            )

        if self.loaded:
            constraint.import_data(self.data)
            constraint.export_data(self.data)

        self.constraints[name] = constraint

    def set_objective(self, objective: Objective, minimize: bool | None = None) -> None:
        "Set the objective function for the environment."
        if self.loaded:
            objective.import_data(self.data)
            objective.export_data(self.data)

        self.objective = objective
        self.minimize = objective.default_minimize if minimize is None else minimize

    def set_instance(
        self,
        instance: InstanceTypes,
        processing_times: ProcessTimeAllowedTypes = None,
        job_instance: InstanceTypes | None = None,
        job_feature: str = "",
        n_parts: int | None = None,
    ) -> None:
        """
        Set the instance data for the environment.

        Parameters:
            instance: InstanceTypes
                The instance data for the scheduling problem, can be a DataFrame or a dictionary.

            processing_times: ProcessTimeAllowedTypes
                The processing times for the tasks, can be a list of dictionaries, a list of lists,
                a list of integers, or a string representing a column in the instance data.

            job_instance: Optional[InstanceTypes]
                The job instance data for the scheduling problem, can be a DataFrame or a
                dictionary. If None, no job instance is set.

            job_ids: Optional[Iterable[int] | str]
                The job IDs for the tasks. If None, job IDs are as the default index of the
                job_instance.

            n_parts: Optional[int]
                The number of parts to split the tasks into. If None, it defaults to 16 if
                preemption is allowed, otherwise 1.
        """
        num_parts: PART_ID = (
            n_parts
            if n_parts is not None  # User-defined number of parts.
            else (
                16
                if self.allow_preemption  # Default number of parts for preemption.
                else 1
            )  # Non-preemptive scheduling.
        )

        task_data = prepare_instance(instance)
        job_data = prepare_instance(job_instance) if job_instance is not None else {}

        parsed_processing_times = self.setup.parse_process_time(
            task_data, processing_times
        )

        self.data = SchedulingData(
            task_data, parsed_processing_times, job_data, job_feature
        )
        self.tasks = Tasks(self.data.job_ids, parsed_processing_times, num_parts)

        for constraint in self.setup.setup_constraints(self.data):
            self.add_constraint(constraint, replace=True)

        for constraint in self.constraints.values():
            constraint.import_data(self.data)
            if not constraint.name.startswith("setup_"):
                constraint.export_data(self.data)

        self.objective.import_data(self.data)
        self.objective.export_data(self.data)

        self.observation_space = self.data.get_gym_space()

        self.loaded = True

    ## Environment state retrieval methods
    def _get_state(self) -> ObsType:
        "Retrieve the current state of the environment from tasks."
        task_data, job_data = self.data.export_state()
        dynamic_task_data, dynamic_job_data = self.tasks.export_state(self.current_time)

        task_data.update(dynamic_task_data)
        job_data.update(dynamic_job_data)

        return task_data, job_data

    def _get_info(self) -> InfoType:
        "Retrieve additional information about the environment."
        return {
            "n_queries": len(self.query_times),
            "current_time": int(self.current_time),
        }

    def _propagate(self) -> None:
        "Propagate the new bounds through the constraints"
        # Currently not sure if this is needed, but keeping it for future reference.

        # for task_id in self.tasks.awaiting_tasks:
        #     task = self.tasks[task_id]
        #     if task.get_start_lb() < self.current_time:
        #         task.set_start_lb(self.current_time)

        for constraint in self.constraints.values():
            constraint.propagate(self.current_time, self.tasks)

        self.tasks.finish_propagation()

    # Environment API methods
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        super().reset(seed=seed)

        if options is not None and "instance" in options:
            instance: InstanceTypes = options["instance"]
            processing_times: ProcessTimeAllowedTypes = options.get(
                "processing_times", None
            )
            job_instance: InstanceTypes | None = options.get("job_instance", None)
            job_feature: str = options.get("job_feature", "")
            n_parts: int | None = options.get("n_parts", None)

            self.set_instance(
                instance, processing_times, job_instance, job_feature, n_parts
            )

        if not self.loaded:
            raise ValueError("Environment not loaded. Please set an instance first.")

        self.schedule.clear()
        self.schedule[-1] = []

        self.current_time = 0
        self.advancing_to = 0
        self.query_times.clear()

        self.tasks.reset()
        for constraint in self.constraints.values():
            constraint.reset(self.tasks)

        self._propagate()

        return self._get_state(), self._get_info()

    ## Environment utility private methods
    def _get_objective(self) -> float:
        "Get the current value of the objective function."
        return float(self.objective.get_current(self.current_time, self.tasks))

    def _is_terminal(self) -> bool:
        "Check if the environment is in a terminal state."
        if self.tasks.awaiting_tasks:
            return False

        for task_id in self.tasks.fixed_tasks:
            if not self.tasks[task_id].is_completed(self.current_time):
                return False

        return True

    def _is_schedule_empty(self) -> bool:
        return not self.schedule[-1] and len(self.schedule) == 1

    def _next_decision_time(self, strict: bool = False) -> TIME:
        "Obtain the next decision time to advance. If strict, only consider future tasks."
        next_time = MAX_INT

        if strict:
            for task_id in self.tasks.awaiting_tasks:
                task = self.tasks[task_id]
                start_lb = task.get_start_lb()

                if self.current_time < start_lb < next_time:
                    next_time = start_lb

            for instruction_time in self.schedule:
                if self.current_time < instruction_time < next_time:
                    next_time = instruction_time

        else:
            for task_id in self.tasks.awaiting_tasks:
                task = self.tasks[task_id]
                start_lb = task.get_start_lb()
                if start_lb < next_time:
                    next_time = start_lb

            for instruction_time in self.schedule:
                if self.current_time <= instruction_time < next_time:
                    next_time = instruction_time

        if self.current_time < self.advancing_to < next_time:
            next_time = self.advancing_to

        if next_time == MAX_INT:
            return self.tasks.get_time_ub()

        if next_time < self.current_time:
            next_time = self.current_time

        return next_time

    def _advance_to_next_instruction(self) -> None:
        "Advance the environment to the next instruction time."
        next_time = self.advancing_to
        for instruction_time in self.schedule:
            if self.current_time <= instruction_time < next_time:
                next_time = instruction_time

        self.current_time = next_time

    def _advance_to_decision_point(self, strict: bool) -> None:
        "Advance the environment to the next decision point."
        next_time = self._next_decision_time(strict)
        self.current_time = next_time

    def _schedule_instruction(
        self, action: str | Instruction, args: tuple[int, ...]
    ) -> None:
        "Add a single instruction to the schedule."
        if action in ("execute", "submit") and 0 < len(args) < 3:
            task = args[0]

            machines = self.tasks[task].machines
            if len(machines) > 1:
                raise ValueError(
                    f"Task {task} has multiple machines assigned: {machines}. "
                    "Please specify the machine to execute on."
                )

            args = (task, int(machines[0]), *args[1:])

        if action == "advance" and len(args) == 0:
            args = (-1,)

        instruction, time = parse_instruction(action, args)

        if time != -1 and time < self.current_time:
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

            signal = instruction.process(self.current_time, self.tasks, self.schedule)

        action: u8 = signal.action
        if not allow_wait and action == Action.WAIT:
            warn(f"{instruction} is not allowed to wait at {self.current_time}.")
            schedule.pop(i)
            return True

        halt = False
        if action & Action.SKIPPED:
            for task_id in self.tasks.fixed_tasks:
                if self.tasks[task_id].is_executing(self.current_time):
                    action = Action.WAIT
                    break

            else:
                halt = True

        if not (action & Action.SKIPPED) and i != -1:
            schedule.pop(i)

        if action & Action.REEVALUATE:
            for task_id in self.tasks.awaiting_tasks:
                self.tasks[task_id].set_start_lb(self.current_time)
            
            for constraint in self.constraints.values():
                constraint.reevaluate(self.current_time, self.tasks)

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

        if action & Action.HALT:
            halt = True

        return halt

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
            if not self.tasks.awaiting_tasks:
                self.current_time = self.tasks.get_time_ub()

            return True

        return self._process_next_instruction()

    def step(
        self,
        action: ActionType = None,
    ) -> tuple[ObsType, float, bool, bool, InfoType]:
        if is_single_action(action):
            single_args = tuple(map(int, action[1:]))
            self._schedule_instruction(action[0], single_args)

        elif action is not None:
            for instruction in action:
                args = tuple(map(int, instruction[1:]))
                self._schedule_instruction(instruction[0], args)

        previous_objective = self._get_objective()

        while True:
            if self._dispatch_instruction():
                break

            if self.render_mode is not None:
                self.render()

        self.query_times.append(self.current_time)

        obs = self._get_state()

        reward = self._get_objective() - previous_objective
        if self.minimize:
            reward *= -1

        truncated = False
        terminal = self._is_terminal()
        info = self._get_info()

        return obs, reward, terminal, truncated, info

    # Environment rendering and representation methods
    def render(self) -> None:
        if self.render_mode == "plot":
            self.renderer.render(self.current_time, self.tasks, self.data)

    def get_entry(self) -> str:
        "Get a string representation of the environment's configuration."
        alpha = self.setup.get_entry()

        beta = (
            ",".join(
                [
                    constraint.get_entry()
                    for name, constraint in self.constraints.items()
                    if not name.startswith("setup_") and constraint.get_entry()
                ]
            )
            + "prmp"
            if self.allow_preemption
            else ""
        )

        gamma = self.objective.get_entry()

        return f"{alpha}|{beta}|{gamma}"

    def __repr__(self) -> str:
        if self.loaded:
            return (
                f"SchedulingEnv({self.get_entry()}, n_tasks={self.tasks.n_tasks}, "
                f"current_time={self.current_time})"
            )

        return f"SchedulingEnv({self.get_entry()}, n_tasks=0)"

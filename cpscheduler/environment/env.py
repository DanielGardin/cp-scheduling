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

from typing import Any, Optional, TypeAlias, Iterable, SupportsInt, TypeGuard, Mapping, cast
from pandas import DataFrame

from gymnasium import Env
from gymnasium.spaces import Dict, Tuple, Text, Box, OneOf

from .common import MAX_INT, ProcessTimeAllowedTypes
from .tasks import Tasks
from .instructions import Instruction, Signal, parse_instruction, Action
from .schedule_setup import ScheduleSetup
from .constraints import Constraint
from .objectives import Objective
from .utils import convert_to_list, is_iterable_type, infer_list_space

from .render import Renderer, PlotlyRenderer

InstanceTypes: TypeAlias = DataFrame | Mapping[str, Iterable[Any]]

SingleAction: TypeAlias = tuple[str | Instruction, *tuple[int, ...]]
ActionType: TypeAlias   = SingleAction | Iterable[SingleAction] | None
ObsType: TypeAlias      = tuple[dict[str, list[Any]], dict[str, list[Any]]]

# Define the action space for the environment
InstructionSpace = Text(max_length=10)
IntSpace = Box(low=0, high=MAX_INT, shape=(), dtype=int) # type: ignore
                                                         # do not want numpy dependency here

ActionSpace = OneOf([
    Tuple([InstructionSpace]),
    Tuple([InstructionSpace, IntSpace]),
    Tuple([InstructionSpace, IntSpace, IntSpace]),
    Tuple([InstructionSpace, IntSpace, IntSpace, IntSpace]),
])

def is_single_action(action: ActionType) -> TypeGuard[tuple[str | Instruction, *tuple[int, ...]]]:
    "Check if the action is a single instruction or a iterable of instructions."
    if not isinstance(action, tuple):
        return False

    if action:
        return isinstance(action[0], str) and is_iterable_type(action[1:], SupportsInt)

    return True


def prepare_instance(instance: InstanceTypes) -> dict[str, list[Any]]:
    "Prepare the instance data to a standard dictionary format."
    features = instance.keys() if isinstance(instance, Mapping) else instance.columns

    return {
        feature: convert_to_list(instance[feature]) for feature in features # type: ignore
    }                                                                       # Dataframes are a pain

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

    """
    # Environment static variables
    setup           : ScheduleSetup
    constraints     : dict[str, Constraint]
    objective       : Objective
    allow_preemption: bool
    minimize        : bool = True

    # Environment dynamic variables
    tasks                 : Tasks
    data                  : dict[str, list[Any]]
    scheduled_instructions: dict[int, list[Instruction]]
    current_time          : int
    query_times           : list[int]

    loaded      : bool
    advancing_to: int

    renderer: Renderer
    _metadata: dict[str, Any]

    def __init__(
        self,
        machine_setup: ScheduleSetup,
        constraints: Optional[Iterable[Constraint]] = None,
        objective  : Optional[Objective]            = None,
        render_mode: Optional[str]                  = None,
        *,
        minimize        : Optional[bool]                    = None,
        allow_preemption: bool                              = False,
        instance         : Optional[InstanceTypes]          = None,
        processing_times: Optional[ProcessTimeAllowedTypes] = None,
        job_instance     : Optional[InstanceTypes]          = None,
        job_ids          : Optional[Iterable[int] | str]    = None,
        n_parts          : Optional[int]                    = None,
    ):
        self.allow_preemption = allow_preemption
        self.loaded = False

        self.setup       = machine_setup

        self.constraints = {}
        if constraints is not None:
            for constraint in constraints:
                self.add_constraint(constraint)

        if objective is None:
            self.objective   = Objective()
        else:
            self.set_objective(objective, minimize)

        self.render_mode = render_mode
        self.scheduled_instructions = {-1: []}

        self.current_time = 0
        self.query_times  = []

        self._metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 50,
        }

        self.action_space      = ActionSpace
        self.observation_space = Tuple([
            Dict({'task_id': Box(low=0, high=MAX_INT, shape=())}),
            Dict({'job_id': Box(low=0, high=MAX_INT, shape=())})]
        ) # Placeholder for observation space

        self.advancing_to = 0

        if instance is not None and processing_times is not None:
            self.set_instance(
                instance,
                processing_times,
                job_instance,
                job_ids,
                n_parts
            )

    def export_environment(self) -> None:
        """
        Export the environment to a JSON file.
        """
        return

    def __repr__(self) -> str:
        if self.loaded:
            return f"SchedulingEnv({self.get_entry()}, n_tasks={len(self.tasks)}, "\
                    "current_time={self.current_time}, loaded=True)"

        return f"SchedulingEnv({self.get_entry()}, not loaded)"

    def add_constraint(self, constraint: Constraint, replace: bool = False) -> None:
        "Add a constraint to the environment."
        name = constraint.name

        if name in self.constraints and not replace:
            raise ValueError(
                f"Constraint with name {name} already exists. Please use a different name."
            )

        self.constraints[name] = constraint

        if self.loaded:
            self.constraints[name].set_tasks(self.tasks)

    def set_objective(self, objective: Objective, minimize: Optional[bool] = None) -> None:
        "Set the objective function for the environment."
        self.objective = objective
        self.minimize  = (
            objective.default_minimize if minimize is None
            else minimize
        )

        if self.loaded:
            self.objective.set_tasks(self.tasks)

    def set_instance(
        self,
        instance         : InstanceTypes,
        processing_times : ProcessTimeAllowedTypes,
        job_instance     : Optional[InstanceTypes] = None,
        job_ids          : Optional[Iterable[int] | str] = None,
        n_parts          : Optional[int] = None,
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
        n_parts = (
            n_parts if n_parts is not None   # User-defined number of parts.
            else 16 if self.allow_preemption # Default number of parts for preemption.
            else 1                           # Non-preemptive scheduling.
        )

        data     = prepare_instance(instance)
        job_data = prepare_instance(job_instance) if job_instance is not None else None

        parsed_processing_times: list[dict[int, int]] = self.setup.parse_process_time(
            data,
            processing_times
        )

        self.tasks = Tasks(
            data,
            parsed_processing_times,
            job_ids,
            job_data,
            n_parts=n_parts
        )

        self.setup.set_tasks(self.tasks)
        for constraint in self.setup.setup_constraints():
            self.add_constraint(constraint, replace=True)

        for constraint in self.constraints.values():
            constraint.set_tasks(self.tasks)

        self.objective.set_tasks(self.tasks)

        task_feature_space = {
            feature: infer_list_space(values)
            for feature, values in self.tasks.data.items()
        }

        job_feature_space = {
            feature: infer_list_space(values)
            for feature, values in self.tasks.jobs_data.items()
        }

        self.observation_space = Tuple([
            Dict(task_feature_space),
            Dict(job_feature_space)
        ])

        self.renderer = PlotlyRenderer(self.tasks, self.setup.n_machines)
        self.loaded   = True

    def get_state(self) -> ObsType:
        "Retrieve the current state of the environment from tasks."
        return self.tasks.get_state(self.current_time)

    def get_info(self) -> dict[str, Any]:
        "Retrieve additional information about the environment."
        return {
            "n_queries": len(self.query_times),
            "current_time": self.current_time,
        }

    def get_objective(self) -> float:
        "Get the current value of the objective function."
        return float(self.objective.get_current(self.current_time))

    def is_terminal(self) -> bool:
        "Check if the environment is in a terminal state."
        return all([task.is_completed(self.current_time) for task in self.tasks])

    def truncate(self) -> bool:
        "Check if the environment is in a truncated state. Legacy method"
        return False

    def propagate(self) -> None:
        "Propagate the new bounds through the constraints"
        for constraint in self.constraints.values():
            constraint.propagate(self.current_time)

    def reset(
        self, *, seed: Optional[int] = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and 'instance' in options and 'processing_times' in options:
            instance         = options['instance']
            processing_times = options['processing_times']
            job_instance     = options.get('job_instance', None)
            job_ids          = options.get('job_ids', None)
            n_parts          = options.get('n_parts', None)

            self.set_instance(
                instance,
                processing_times,
                job_instance,
                job_ids,
                n_parts
            )


        if not self.loaded:
            raise ValueError("Environment not loaded. Please set an instance first.")

        self.scheduled_instructions.clear()
        self.scheduled_instructions[-1] = []

        self.current_time = 0
        self.advancing_to = 0
        self.query_times.clear()

        self.tasks.reset()
        for constraint in self.constraints.values():
            constraint.reset()

        self.propagate()

        return self.get_state(), self.get_info()

    def instruction_times(self, strict: bool = False) -> list[int]:
        "Return the next instruction times in the schedule."
        if strict:
            instruction_times = [
                instruction_time for instruction_time in self.scheduled_instructions
                if instruction_time > self.current_time
            ]

        else:
            instruction_times = list(self.scheduled_instructions.keys())
            instruction_times.remove(-1)

        return instruction_times

    def next_decision_time(self, strict: bool = False) -> int:
        "Obtain the next decision time to advance. If strict, only consider future tasks."
        if strict:
            operation_points = [
                task.get_start_lb() for task in self.tasks
                if task.is_awaiting() and task.get_start_lb() > self.current_time
            ]

        else:
            operation_points = [
                task.get_start_lb() for task in self.tasks
                if task.is_awaiting()
            ]

        if not operation_points and strict:
            operation_points = [
                task.get_end() for task in self.tasks
                if task.is_executing(self.current_time) and task.get_end() > self.current_time
            ]

        instruction_times = self.instruction_times(strict)

        time = MAX_INT
        if operation_points:
            min_operation_time = min(operation_points)
            time = min(time, min_operation_time)

        if instruction_times:
            min_instruction_time = min(instruction_times)
            time = min(time, min_instruction_time)

        if self.current_time < self.advancing_to:
            time = min(time, self.advancing_to)

        if time == MAX_INT:
            return self.tasks.get_time_ub()

        return max(time, self.current_time)

    def advance_to(self, time: int) -> None:
        "Advance the environment to a specific time."
        self.current_time = time

    def advance_to_next_instruction(self) -> None:
        "Advance the environment to the next instruction time."
        min_instruction_time = min(self.instruction_times(), default=self.advancing_to)

        time = min(min_instruction_time, self.advancing_to)
        self.advance_to(time)

    def advance_decision_point(self) -> None:
        "Advance the environment to the next decision point."
        time = self.next_decision_time(False)
        self.advance_to(time)

    def skip_to_next_decision_point(self) -> None:
        "Skip to the next decision point in the future."
        time = self.next_decision_time(True)
        self.advance_to(time)

    def schedule_instruction(self, action: str | Instruction, args: tuple[int, ...]) -> None:
        "Add a single instruction to the schedule."
        if not self.setup.parallel and action in ("execute", "submit"):
            if len(args) == 2:
                task, time = args

                args = (task, self.setup.get_machine(task), time)


            elif len(args) == 1:
                task, = args
                args = (task, self.setup.get_machine(task))

        if action == "advance":
            if len(args) == 0:
                args = (-1,)

        instruction, time = parse_instruction(action, args)

        if time != -1 and time < self.current_time:
            warn(
                f"Scheduled instruction {instruction} with arguments {args} is in the past. \
                It will be executed immediately."
            )
            time = self.current_time

        if time not in self.scheduled_instructions:
            self.scheduled_instructions[time] = []

        self.scheduled_instructions[time].append(instruction)


    def resolve_schedule(self, time: int = -1, allow_wait: bool = True) -> tuple[bool, bool]:
        """
        Process the next instruction in the schedule at the given time.
        If time is -1, use the current time and if allow_wait is False,
        it halts the environment when an instruction is requested to be waited.
        """
        instruction = Instruction()
        signal      = Signal(Action.SKIPPED)

        i = -1
        schedule = self.scheduled_instructions[time]

        while signal.action & Action.SKIPPED and i+1 < len(schedule):
            i += 1
            instruction = schedule[i]

            signal = instruction.process(
                self.current_time,
                self.tasks,
                self.scheduled_instructions
            )

        action = signal.action
        halt, change = False, False

        if not allow_wait and action == Action.WAIT:
            warn(
                f"{instruction} is not allowed to wait at {self.current_time}."
            )
            schedule.pop(i)
            return True, False

        if action & Action.SKIPPED:
            for task in self.tasks:
                if task.is_executing(self.current_time):
                    action |= Action.PROPAGATE | Action.ADVANCE_NEXT
                    break

            else:
                if i != -1: schedule.pop(i) 
                else:     halt = True

                return halt, change

        elif i != -1:
            schedule.pop(i)

        if action & Action.REEVALUATE:
            for task in self.tasks:
                if not task.is_fixed():
                    task.set_start_lb(self.current_time)

        if action & Action.PROPAGATE:
            self.propagate()
            change = True

        if action & Action.ADVANCE:
            previous_time = self.current_time
            if signal.param > 0:
                self.advancing_to = signal.param
                self.advance_to_next_instruction()
                self.propagate()
                change = True

            else:
                self.advance_decision_point()

            change = self.current_time != previous_time

        if action & Action.ADVANCE_NEXT:
            self.skip_to_next_decision_point()
            change = True

        if action & Action.RAISE:
            warn(f"Error in instruction {instruction} at time {self.current_time}: {signal.info}")

        if action & Action.HALT:
            halt = True

        return halt, change

    def execute_next_instruction(self) -> tuple[bool, bool]:
        "Dispatch the next instruction in the schedule depending on the current state."
        if self.current_time in self.scheduled_instructions:
            halt, change = self.resolve_schedule(self.current_time, False)

            if not self.scheduled_instructions[self.current_time]:
                self.scheduled_instructions.pop(self.current_time)

        elif self.current_time < self.advancing_to:
            self.propagate()
            self.advance_to_next_instruction()
            halt, change = False, True

        elif not self.scheduled_instructions[-1] and len(self.scheduled_instructions) == 1:
            halt, change = True, False

            if all(task.is_fixed() for task in self.tasks):
                end_time = self.tasks.get_time_ub()
                self.advance_to(end_time)

                change = end_time != self.current_time

        else:
            halt, change = self.resolve_schedule()

        return halt, change

    def step(
        self,
        action: ActionType = None,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        if is_single_action(action):
            self.schedule_instruction(action[0], action[1:])

        elif action is not None:
            action = cast(Iterable[SingleAction], action)
            for instruction in action:
                self.schedule_instruction(instruction[0], instruction[1:])

        previous_objective = self.get_objective()

        stop = False
        while not stop:
            stop, change = self.execute_next_instruction()
            if change and self.render_mode is not None:
                self.render()

        self.query_times.append(self.current_time)

        obs = self.get_state()

        reward = self.get_objective() - previous_objective
        if self.minimize:
            reward *= -1

        info = self.get_info()

        return obs, reward, self.is_terminal(), self.truncate(), info

    def render(self) -> None:
        if self.render_mode == "plot":
            self.renderer.render(self.current_time)


    def get_entry(self) -> str:
        "Get a string representation of the environment's configuration."
        alpha = self.setup.get_entry()

        beta = ", ".join([
            constraint.get_entry()
            for name, constraint in self.constraints.items()
            if not name.startswith("setup_") and constraint.get_entry()
        ]) + "prmp" if self.allow_preemption else ""

        gamma = self.objective.get_entry()

        return f"{alpha}|{beta}|{gamma}"

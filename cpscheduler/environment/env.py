from typing import Any, Optional, TypeAlias, Iterable, SupportsInt, TypeGuard, Literal
from numpy.typing import ArrayLike
from pandas import DataFrame

from numpy import int64

from warnings import warn

from gymnasium import Env
from gymnasium.spaces import Dict, Tuple, Text, Box, OneOf

from .tasks import Tasks, status_str
from .instructions import Instruction, Signal, parse_instruction, Action
from .schedule_setup import ScheduleSetup
from .constraints import Constraint
from .objectives import Objective
from .utils import MAX_INT, convert_to_list, is_iterable_type, is_dict, infer_list_space

from .render import Renderer, PlotlyRenderer

TaskAllowedTypes: TypeAlias        = DataFrame | dict[str, ArrayLike]
ProcessTimeAllowedTypes: TypeAlias = ArrayLike | Iterable[int] | str

SingleAction: TypeAlias = tuple[str | Instruction, *tuple[int, ...]]
ActionType: TypeAlias   = SingleAction | Iterable[SingleAction] | None

# Define the action space for the environment
InstructionSpace = Text(max_length=10)
IntSpace = Box(low=0, high=MAX_INT, shape=(), dtype=int64)

ActionSpace = OneOf([
    Tuple([InstructionSpace]),
    Tuple([InstructionSpace, IntSpace]),
    Tuple([InstructionSpace, IntSpace, IntSpace]),
    Tuple([InstructionSpace, IntSpace, IntSpace, IntSpace]),
])

def is_single_action(
    action: ActionType,
) -> TypeGuard[tuple[str | Instruction, *tuple[int, ...]]]:
    if not isinstance(action, tuple):
        return False

    if action:
        return isinstance(action[0], str) and is_iterable_type(action[1:], SupportsInt)

    return True


class SchedulingCPEnv(Env[dict[str, list[Any]], ActionType]):
    metadata: dict[str, Any] = {
        "render_modes": ["plot", "rgb_array_list"],
    }

    # Environment static variables
    constraints: dict[str, Constraint]
    objective: Objective

    # Environment dynamic variables
    tasks: Tasks
    data: dict[str, list[Any]]
    scheduled_instructions: dict[int, list[Instruction]]
    current_time: int
    query_times: list[int]

    advancing_to: int

    renderer: Renderer

    # Functions with interaction with the user can be slower.
    def __init__(
        self,
        machine_setup: Optional[ScheduleSetup] = None,
        constraints: Optional[dict[str, Constraint] | Iterable[Constraint]] = None,
        objective: Optional[Objective] = None,
        instance: Optional[TaskAllowedTypes] = None,
        render_mode: Optional[str] = None,
        *,
        n_machines: int = 1,
        allow_preemption: bool = False,
        processing_times: ProcessTimeAllowedTypes = "processing_time",
        jobs: Optional[Iterable[int] | str] = None,
    ):
        self.setup = (
            machine_setup if machine_setup is not None else ScheduleSetup(n_machines)
        )
        self.allow_preemption = allow_preemption

        self.constraints = {}
        self.objective = Objective()

        self.scheduled_instructions = {-1: []}

        self.current_time = 0
        self.query_times  = []

        self.advancing_to = 0

        self.action_space = ActionSpace

        if is_iterable_type(constraints, Constraint):
            for constraint in constraints:
                self.add_constraint(constraint)

        elif is_dict(constraints, str, Constraint):
            for name, constraint in constraints.items():
                self.add_constraint(constraint, name)

        if objective is not None:
            self.set_objective(objective)

        if instance is not None:
            self.set_instance(instance, processing_times=processing_times, jobs=jobs)
        
        self.render_mode = render_mode

    def load_configuration(self, config: dict[str, Any]) -> None:
        # TODO: Implement the loading of the configuration
        pass

    def add_constraint(
        self, constraint: Constraint, name: Optional[str] = None
    ) -> None:
        name = name if name is not None else constraint.__class__.__name__

        self.constraints[name] = constraint

    def set_objective(self, objective: Objective) -> None:
        self.objective = objective

    def set_instance(
        self,
        instance: TaskAllowedTypes,
        processing_times: ProcessTimeAllowedTypes = "processing_time",
        jobs: Optional[Iterable[int] | str] = None,
        n_parts: Optional[
            int
        ] = None,  # if we allow preemption, we must define a maximum number of splits
    ) -> None:
        if n_parts is None:
            n_parts = 16 if self.allow_preemption else 1

        features = instance.keys() if isinstance(instance, dict) else instance.columns
        data = {feature: convert_to_list(instance[feature]) for feature in features}

        if isinstance(processing_times, str):
            processing_times = data[processing_times]

        assert is_iterable_type(processing_times, SupportsInt)
        if jobs is None:
            jobs = [i for i, _ in enumerate(processing_times)]

        elif isinstance(jobs, str):
            jobs = data[jobs]

        self.tasks = Tasks(data, n_parts)

        for processing_time, job in zip(processing_times, jobs):
            self.tasks.add_task(int(processing_time), job)

        self.setup.set_tasks(self.tasks)
        for constraint in self.setup.setup_constraints():
            name = f"_setup_{constraint.__class__.__name__}"
            self.add_constraint(constraint, name)

        for constraint in self.constraints.values():
            constraint.set_tasks(self.tasks)

        self.objective.set_tasks(self.tasks)

        self.observation_space = Dict(
            {
                "task_id": Box(
                    low=0, high=len(self.tasks), shape=(len(self.tasks),), dtype=int64
                ),
                **{feature: infer_list_space(data[feature]) for feature in data.keys()},
                "remaining_time": Box(
                    low=0, high=MAX_INT, shape=(len(self.tasks),), dtype=int64
                ),
                "status": Text(max_length=1, charset=frozenset(status_str.values())),
            }
        )

        self.renderer = PlotlyRenderer(self.tasks, self.setup.n_machines)

    def get_state(self) -> dict[str, list[Any]]:
        return self.tasks.get_state(self.current_time)

    def get_info(self) -> dict[str, Any]:
        return {
            "n_queries": len(self.query_times),
            "current_time": self.current_time,
        }

    def get_objective(self) -> float:
        return float(self.objective.get_current(self.current_time))

    def is_terminal(self) -> bool:
        return all([task.is_completed(self.current_time) for task in self.tasks])

    def truncate(self) -> bool:
        return False

    def propagate(self) -> None:
        for constraint in self.constraints.values():
            constraint.propagate(self.current_time)

    def reset(
        self, *, seed: Optional[int] = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        super().reset(seed=seed)

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
        if strict:
            instruction_times = [
                instruction_time
                for instruction_time in self.scheduled_instructions.keys()
                if instruction_time > self.current_time
            ]

        else:
            instruction_times = list(self.scheduled_instructions.keys())
            instruction_times.remove(-1)

        return instruction_times

    def next_decision_time(self, strict: bool = False) -> int:
        if strict:
            operation_points = [
                task.get_start_lb() for task in self.tasks
                if task.is_awaiting(self.current_time) and task.get_start_lb() > self.current_time
            ]

        else:
            operation_points = [
                task.get_start_lb() for task in self.tasks
                if task.is_awaiting(self.current_time)
            ]

        instruction_times = self.instruction_times(strict)

        time = MAX_INT
        if operation_points:
            time = min(time, min(operation_points))

        if instruction_times:
            time = min(time, min(instruction_times))

        if self.current_time < self.advancing_to:
            time = min(time, self.advancing_to)

        if time == MAX_INT:
            return self.tasks.get_time_ub()

        return time

    def advance_to(self, time: int) -> None:
        self.current_time = time

    def advance_to_next_instruction(self) -> None:
        min_instruction_time = min(self.instruction_times(), default=self.advancing_to)

        time = min(min_instruction_time, self.advancing_to)
        self.advance_to(time)

    def advance_decision_point(self) -> None:
        time = self.next_decision_time(False)
        self.advance_to(time)

    def skip_to_next_decision_point(self) -> None:
        time = self.next_decision_time(True)
        self.advance_to(time)

    def schedule_instruction(
        self, action: str | Instruction, args: tuple[int, ...]
    ) -> None:
        if not self.setup.parallel_machines and (action == "execute" or action == "submit"):
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
        instruction = Instruction()
        signal      = Signal(Action.SKIPPED)

        i = -1
        schedule = self.scheduled_instructions[time]
        while signal.action & Action.SKIPPED and i+1 < len(schedule):
            i += 1
            instruction = schedule[i]

            signal = instruction.process(
                self.current_time, self.tasks, self.scheduled_instructions
            )

        action       = signal.action
        halt, change = False, False

        if action & Action.SKIPPED:
            halt = True
            for task in self.tasks:
                if task.is_executing(self.current_time):
                    action = Action.PROPAGATE | Action.ADVANCE_NEXT

                    halt = self.current_time >= self.tasks.get_time_ub()
                    break
        
        else:
            schedule.pop(i)
        
        if action & Action.REEVALUATE:
            for task in self.tasks:
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
        if self.current_time in self.scheduled_instructions:
            halt, change = self.resolve_schedule(self.current_time, False)

            if not self.scheduled_instructions[self.current_time]:
                self.scheduled_instructions.pop(self.current_time)

        elif self.current_time < self.advancing_to:
            self.propagate() 
            self.advance_to_next_instruction()
            halt, change = False, True

        elif not self.scheduled_instructions[-1] and len(self.scheduled_instructions) == 1:
            # for task in self.tasks:
            #     if task.is_available(self.current_time):
            #         return True, False

            # self.propagate()
            # self.skip_to_next_decision_point()
            halt, change = True, False

        else:
            halt, change = self.resolve_schedule()

        return halt, change

    def step(
        self,
        action: ActionType = None,
    ) -> tuple[dict[str, list[Any]], float, bool, bool, dict[str, Any]]:
        if is_single_action(action):
            self.schedule_instruction(action[0], action[1:])

        elif action is not None:
            assert is_iterable_type(action, tuple)
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
        if self.objective.direction == "minimize":
            reward *= -1

        info = self.get_info()

        return obs, reward, self.is_terminal(), self.truncate(), info

    def render(self) -> None:
        if self.render_mode == "plot":
            self.renderer.plot(self.current_time)
        
        elif self.render_mode == "rgb_array_list":
            self.renderer.build_gantt(self.current_time)

    def export(self, filename: Optional[str] = None) -> tuple[str, str]:
        data_file = '\n'.join([
            self.tasks.export_data(),
            self.setup.export_data(),
            *[constraint.export_data() for constraint in self.constraints.values()],
            self.objective.export_data()
        ])

        model_file = '\n'.join([
            self.tasks.export_model(),
            self.setup.export_model(),
            *[constraint.export_model() for constraint in self.constraints.values()],
            self.objective.export_model()
        ])

        if filename is not None:
            with open(f"{filename}.dzn", "w") as file:
                file.write(data_file)

            with open(f"{filename}.mzn", "w") as file:
                file.write(model_file)

        return data_file, model_file

    def get_entry(self) -> str:
        alpha = self.setup.get_entry()
        beta = ", ".join(
            [
                constraint.get_entry()
                for name, constraint in self.constraints.items()
                if not name.startswith("_setup_") and constraint.get_entry()
            ]
        )
        gamma = self.objective.get_entry()

        return f"{alpha} | {beta} | {gamma}"
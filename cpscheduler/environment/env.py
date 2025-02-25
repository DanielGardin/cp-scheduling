from typing import Any, Optional, ClassVar, TypeAlias, Iterable, SupportsInt, TypeGuard
from minizinc import Result, Instance
from numpy.typing import ArrayLike
from pandas import DataFrame

from numpy import int64

from warnings import warn
from datetime import timedelta

from gymnasium import Env
from gymnasium.spaces import Dict, Tuple, Text, Box, OneOf

from collections import deque

from .tasks import Tasks, status_str
from .schedule_setup import ScheduleSetup
from .constraints import Constraint
from .objectives import Objective
from .utils import MAX_INT, convert_to_list, is_iterable_type, is_dict, infer_list_space

from .instructions import Instruction, Signal, Instructions
from . import instructions


TaskAllowedTypes: TypeAlias = DataFrame | dict[str, ArrayLike]
ProcessTimeAllowedTypes: TypeAlias = ArrayLike | Iterable[int] | str
ActionType: TypeAlias = (
    Iterable[tuple[str | Instruction, *tuple[int, ...]]]
    | tuple[str | Instruction, *tuple[int, ...]]
    | None
)

InstructionSpace = Text(max_length=1, charset=frozenset(Instructions))
IntSpace = Box(low=0, high=MAX_INT, shape=(), dtype=int64)
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
) -> TypeGuard[tuple[str | Instruction, *tuple[int, ...]]]:
    return isinstance(action, tuple) and isinstance(action[0], str)


def parse_args(
    instruction_name: str,
    args: tuple[int, ...],
    n_required: int,
) -> tuple[tuple[int, ...], int]:
    if len(args) == n_required:
        return args, -1

    elif len(args) == n_required + 1:
        return args[:-1], args[-1]

    else:
        raise ValueError(
            f"Expected {n_required} or {n_required + 1} arguments for instruction {instruction_name} , got {len(args)}."
        )


class SchedulingCPEnv(Env[dict[str, list[Any]], ActionType]):
    # Environment static variables
    constraints: dict[str, Constraint]
    objective: Objective

    # Environment dynamic variables
    tasks: Tasks
    data: dict[str, list[Any]]
    scheduled_instructions: dict[int, deque[Instruction]]
    current_time: int
    n_queries: int

    advancing_to: int

    # Functions with interaction with the user can be slower.
    def __init__(
        self,
        machine_setup: Optional[ScheduleSetup] = None,
        constraints: Optional[dict[str, Constraint] | Iterable[Constraint]] = None,
        objective: Optional[Objective] = None,
        instance: Optional[TaskAllowedTypes] = None,
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

        self.scheduled_instructions = {-1: deque()}

        self.current_time = 0
        self.n_queries = 0

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

    def get_state(self) -> dict[str, list[Any]]:
        return self.tasks.get_state(self.current_time)

    def get_info(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "current_time": self.current_time,
        }

    def get_objective(self) -> float:
        return float(self.objective.get_current(self.current_time))

    def is_terminal(self) -> bool:
        return all([task.is_completed(self.current_time) for task in self.tasks])

    def truncate(self) -> bool:
        return False

    def update_state(self) -> None:
        for constraint in self.constraints.values():
            constraint.propagate(self.current_time)

    def reset(
        self, *, seed: Optional[int] = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        super().reset(seed=seed)

        self.scheduled_instructions.clear()
        self.scheduled_instructions[-1] = deque()

        self.current_time = 0
        self.n_queries = 0
        self.advancing_to = 0

        self.tasks.reset()
        for constraint in self.constraints.values():
            constraint.reset()

        self.update_state()

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
                task.get_start_lb()
                for task in self.tasks
                if task.is_awaiting(self.current_time)
                and task.get_start_lb() > self.current_time
            ]

        else:
            operation_points = [
                task.get_start_lb()
                for task in self.tasks
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
        self.update_state()

    def advance_to_next_instruction(self) -> None:
        time = max(
            min(self.instruction_times(), default=self.advancing_to), self.current_time
        )
        self.advance_to(time)

    def advance_decision_point(self) -> None:
        time = self.next_decision_time(False)
        self.advance_to(time)

    def skip_to_next_decision_point(self) -> None:
        time = self.next_decision_time(True)
        self.advance_to(time)

    def schedule_instruction(
        self, instruction: str | Instruction, args: tuple[int, ...]
    ) -> None:
        match instruction:
            case "execute":
                if self.setup.parallel_machines:
                    (task_id, machine), time = parse_args(instruction, args, 2)

                else:
                    (task_id,), time = parse_args(instruction, args, 1)
                    machine = 0

                instruction = instructions.Execute(task_id, machine)

            case "submit":
                if self.setup.parallel_machines:
                    (task_id, machine), time = parse_args(instruction, args, 2)

                else:
                    (task_id,), time = parse_args(instruction, args, 1)
                    machine = self.setup.get_machine(task_id)

                instruction = instructions.Submit(task_id, machine)

            case "pause":
                (task_id,), time = parse_args(instruction, args, 1)
                instruction = instructions.Pause(task_id)

            case "complete":
                (task_id,), time = parse_args(instruction, args, 1)
                instruction = instructions.Complete(task_id)

            case "advance":
                (to_time,), time = parse_args(instruction, args, 1)
                instruction = instructions.Advance(to_time)

            case "query":
                args, time = parse_args(instruction, args, 0)
                instruction = instructions.Query()

            case "clear":
                args, time = parse_args(instruction, args, 0)
                instruction = instructions.Clear()

            case Instruction():
                args, time = parse_args(instruction.name, args, 0)

            case _:
                raise ValueError(f"Instruction {instruction} not recognized.")

        if time != -1 and time < self.current_time:
            warn(
                f"Scheduled instruction {instruction} with arguments {args} is in the past. \
                It will be executed immediately."
            )
            time = self.current_time

        if time not in self.scheduled_instructions:
            self.scheduled_instructions[time] = deque()

        self.scheduled_instructions[time].append(instruction)

    def execute_next_instruction(self) -> bool:
        if self.current_time in self.scheduled_instructions:
            instruction = self.scheduled_instructions[self.current_time].popleft()

            if not self.scheduled_instructions[self.current_time]:
                del self.scheduled_instructions[self.current_time]

            signal, args = instruction.process(
                self.current_time, self.tasks, self.scheduled_instructions
            )

            if signal.is_failure():
                if isinstance(args, str):
                    warn(f"Instruction {instruction} failed with message: {args}.")

                else:
                    warn(
                        f"Scheduled instruction {instruction} could not be executed at time {self.current_time}."
                    )

                return True

        elif self.current_time < self.advancing_to:
            self.advance_to_next_instruction()
            return False

        elif (
            not self.scheduled_instructions[-1]
            and len(self.scheduled_instructions) == 1
        ):
            # Force query when no instructions are left
            return True

        else:
            for i, instruction in enumerate(self.scheduled_instructions[-1]):
                signal, args = instruction.process(
                    self.current_time, self.tasks, self.scheduled_instructions
                )

                if signal == Signal.Skip:
                    continue

                elif signal != Signal.Pending:
                    del self.scheduled_instructions[-1][i]
                    break

            else:
                for task in self.tasks:
                    if task.is_executing(self.current_time):
                        self.skip_to_next_decision_point()
                        return self.current_time >= self.tasks.get_time_ub()

                return True

        if signal == Signal.Finish:
            self.advance_decision_point()

        elif signal == Signal.Pending:
            self.skip_to_next_decision_point()

        elif signal == Signal.Halt:
            return True

        elif signal == Signal.Advance:
            assert isinstance(args, int)

            self.advancing_to = args
            self.skip_to_next_decision_point()

        elif signal == Signal.Error:
            warn(f"Instruction {instruction.name} failed with message: {args}.")
            return True

        return False

    def step(
        self,
        action: ActionType,
    ) -> tuple[dict[str, list[Any]], float, bool, bool, dict[str, Any]]:

        act_instructions: Iterable[tuple[str | Instruction, *tuple[int, ...]]]
        if is_single_action(action):
            act_instructions = [action]

        elif action is None:
            act_instructions = []

        else:
            assert is_iterable_type(action, tuple)
            act_instructions = action

        for instruction_args in act_instructions:
            self.schedule_instruction(instruction_args[0], instruction_args[1:])

        previous_objective = self.get_objective()

        stop = False
        while not stop:
            stop = self.execute_next_instruction()

        self.n_queries += 1

        obs = self.get_state()

        reward = self.get_objective() - previous_objective
        if self.objective.direction == "minimize":
            reward *= -1

        info = self.get_info()

        return obs, reward, self.is_terminal(), self.truncate(), info

    def export(self, filename: str) -> None:
        with open(f"{filename}.mzn", "w") as file:
            file.write(self.tasks.export_model())
            file.write(self.setup.export_model())

            for constraint in self.constraints.values():
                file.write(constraint.export_model())

            file.write(self.objective.export_model())

        with open(f"{filename}.dzn", "w") as file:
            file.write(self.tasks.export_data())
            file.write(self.setup.export_data())

            for constraint in self.constraints.values():
                file.write(constraint.export_data())

            file.write(self.objective.export_data())

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

"""
instructions.py

This module defines the instructions that can be executed in the scheduling environment.
Instructions are used to control the execution of tasks, manage their states, and interact
with the scheduler.
"""

from typing import Final, TypeAlias
from collections.abc import Iterable
from typing_extensions import Unpack, TypeIs

from dataclasses import dataclass

from mypy_extensions import mypyc_attr, u8

from cpscheduler.environment._common import TASK_ID, TIME, MACHINE_ID, Int
from cpscheduler.environment.state import ScheduleState

SingleAction: TypeAlias = tuple["str | Instruction", Unpack[tuple[Int, ...]]]
ActionType: TypeAlias = SingleAction | Iterable[SingleAction] | None


def is_single_action(
    action: ActionType,
) -> TypeIs[SingleAction]:
    "Check if the action is a single instruction or a iterable of instructions."
    if not isinstance(action, tuple):
        return False

    return isinstance(action[0], str) and all(
        isinstance(arg, Int) for arg in action[1:]
    )


# Flags are not supported by mypyc yet
class Action:
    "Flags for possible actions made by the scheduler in response to an instruction."

    # Tell the scheduler the instruction was skiped, the default behavior is to remove
    # the instruction after processing it, but this flag disables that
    SKIPPED: Final[u8] = 1

    # The scheduler should reevaluate the current bounds (used when preempting tasks)
    REEVALUATE: Final[u8] = 2

    # The scheduler should propagate the constraints further
    PROPAGATE: Final[u8] = 4

    # The scheduler should advance the time
    ADVANCE_TO: Final[u8] = 8

    # The scheduler should advance to the next decision point
    ADVANCE_NEXT: Final[u8] = 16

    # The scheduler should raise an exception
    RAISE: Final[u8] = 32

    # The scheduler should stop processing instructions
    HALT: Final[u8] = 64

    DONE: Final[u8] = PROPAGATE
    ERROR: Final[u8] = RAISE | HALT
    WAIT: Final[u8] = SKIPPED | ADVANCE_NEXT


@dataclass
class Signal:
    "Action signal with additional parameters."

    action: u8
    time: TIME = 0
    info: str = ""


@mypyc_attr(allow_interpreted_subclasses=True)
class Instruction:
    """
    Base class for all instructions in the scheduling environment.

    Instructions are used to control the execution of tasks, manage their states, and interact
    with the scheduler. Each instruction has a name and a method to process it.

    You can create custom instructions by subclassing this class and implementing the
    `process` method.
    Caution: Instructions directly manipulate the state of tasks and the scheduler, so
    new ones should be implemented carefully.
    """

    def process(
        self,
        current_time: TIME,
        state: ScheduleState,
        scheduled_instructions: dict[TIME, list["Instruction"]],
    ) -> Signal:
        "Process the instruction at the given current time."
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Execute(Instruction):
    "Executes a task on a specific machine. If the task cannot be executed, it is waited for."

    def __init__(
        self,
        id: TASK_ID,
        machine: MACHINE_ID = -1,
        job_oriented: bool = False,
        wait: bool = False,
    ):
        self.id = id
        self.machine = machine
        self.job_oriented = job_oriented
        self.wait = wait

    def __repr__(self) -> str:
        instruction_name = "Submit" if self.wait else "Execute"
        orientation = "job" if self.job_oriented else "task"
        
        if self.machine != -1:
            return f"{instruction_name}({orientation}={self.id}, machine={self.machine})"

        return f"{instruction_name}({orientation}={self.id})"

    def process(
        self,
        current_time: TIME,
        state: ScheduleState,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        if self.job_oriented:
            executed_task = state.execute_job(self.id, current_time, self.machine)

            if executed_task >= 0:
                return Signal(Action.DONE)

            if all(task.fixed for task in state.jobs[self.id]):
                return Signal(
                    Action.RAISE,
                    info=f"Every task in Job {self.id} has been executed.",
                )

        else:
            execute = state.execute_task(self.id, current_time, self.machine)

            if execute:
                return Signal(Action.DONE)

            if state.tasks[self.id].fixed:
                return Signal(
                    Action.RAISE,
                    info=f"Task {self.id} was/is already executed.",
                )

            if self.machine > 0 and not self.machine in state.tasks[self.id].machines:
                return Signal(
                    Action.RAISE,
                    info=f"Task {self.id} cannot be executed on machine {self.machine}.",
                )

        return Signal(
            Action.WAIT if self.wait else Action.SKIPPED,
        )


class Submit(Execute):
    "Submits a task to a specific machine. If the task cannot be executed, it is waited for."

    def __init__(
        self,
        id: TASK_ID,
        machine: MACHINE_ID = -1,
        job_oriented: bool = False,
    ):
        super().__init__(id, machine, job_oriented, wait=True)


class Pause(Instruction):
    "Pauses a task if it is currently executing. Can only be used in preemptive scheduling."

    def __init__(
        self,
        id: TASK_ID,
        job_oriented: bool = False,
    ):
        self.id = id
        self.job_oriented = job_oriented

    def __repr__(self) -> str:
        unit = "job" if self.job_oriented else "task"

        return f"Pause({unit}={self.id})"

    def process(
        self,
        current_time: TIME,
        state: ScheduleState,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        if self.job_oriented:
            paused_job = state.pause_job(self.id, current_time)

            if paused_job >= 0:
                return Signal(Action.DONE | Action.REEVALUATE)

            job = state.jobs[self.id]

            if all(task.is_completed(current_time) for task in job):
                return Signal(
                    Action.RAISE,
                    info=f"Job {self.id} cannot be paused. It has been already completed.",
                )

        else:
            can_pause = state.pause_task(self.id, current_time)

            if can_pause:
                return Signal(Action.DONE | Action.REEVALUATE)

            task = state.tasks[self.id]

            if not task.preemptive:
                return Signal(
                    Action.RAISE,
                    info=f"Task {self.id} cannot be paused. Preemption is not allowed.",
                )

            elif state.tasks[self.id].is_completed(current_time):
                return Signal(
                    Action.RAISE,
                    info=f"Task {self.id} cannot be paused. It has been already completed.",
                )

        return Signal(Action.WAIT)


class Resume(Instruction):
    "Resumes a paused task in the same machine it was executing before being paused."

    def __init__(
        self,
        id: TASK_ID,
        job_oriented: bool = False
    ):
        self.id = id
        self.job_oriented = job_oriented

    def __repr__(self) -> str:
        orientation = "job" if self.job_oriented else "task"

        return f"Resume({orientation}={self.id})"

    def process(
        self,
        current_time: TIME,
        state: ScheduleState,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        if self.job_oriented:
            job_tasks = state.jobs[self.id]

            for task in job_tasks:
                if task.is_paused(current_time):
                    last_machine = task.get_assignment()
                    execute = state.execute_task(task.task_id, current_time, last_machine)

                    if execute:
                        return Signal(Action.DONE)


            if all(task.fixed for task in job_tasks):
                return Signal(
                    Action.RAISE,
                    info=f"Every task in Job {self.id} has been executed.",
                )

        else:
            task = state.tasks[self.id]

            if not task.is_paused(current_time):
                return Signal(
                    Action.RAISE,
                    info=f"Task {self.id} is not paused and cannot be resumed.",
                )
            
            last_machine = task.get_assignment()
            execute = state.execute_task(self.id, current_time, last_machine)

            if execute:
                return Signal(Action.DONE)

            if state.tasks[self.id].fixed:
                return Signal(
                    Action.RAISE,
                    info=f"Task {self.id} was/is already executed.",
                )

        return Signal(Action.WAIT)


class Complete(Instruction):
    "Advances the current time to the end of an executing task."

    def __init__(self, task_id: TASK_ID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return f"Complete(task={self.task_id})"

    def process(
        self,
        current_time: TIME,
        state: ScheduleState,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        task = state.tasks[self.task_id]
        if task.is_executing(current_time):
            return Signal(Action.ADVANCE_TO, task.get_end())

        if task.is_completed(current_time):
            return Signal(Action.ERROR, info=f"Task {self.task_id} already terminated")

        return Signal(Action.WAIT)


class Advance(Instruction):
    "Advances the current time by a specified amount or to the next decision point if not specified."

    def __init__(self, dt: TIME = -1) -> None:
        self.dt = dt

    def __repr__(self) -> str:
        if self.dt < 0:
            return "Advance()"

        return f"Advance(dt={self.dt})"

    def process(
        self,
        current_time: TIME,
        state: ScheduleState,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        if self.dt < 0:
            return Signal(Action.ADVANCE_NEXT)

        return Signal(Action.ADVANCE_TO, current_time + self.dt)


class Query(Instruction):
    "When processed, halts the environment and returns its current state."

    def process(
        self,
        current_time: TIME,
        state: ScheduleState,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        return Signal(Action.HALT)


class Clear(Instruction):
    "Clears all upcoming instructions and resets the schedule."

    def process(
        self,
        current_time: TIME,
        state: ScheduleState,
        scheduled_instructions: dict[TIME, list[Instruction]],
    ) -> Signal:
        scheduled_instructions.clear()
        scheduled_instructions[-1] = []

        return Signal(Action.DONE)


def parse_args(
    args: tuple[int, ...],
    output_size: int,
) -> tuple[int, ...]:
    "Parse the raw instruction arguments into required and optional arguments."
    return args + (-1,) * (output_size - len(args))


def parse_instruction(
    action: str | Instruction, args: tuple[int, ...], state: ScheduleState
) -> tuple[Instruction, TIME]:
    "Parse raw instruction arguments into an Instruction object and the scheduled time."
    if isinstance(action, Instruction):
        (time,) = parse_args(args, 1)
        instruction = action

    else:
        is_execute = action.startswith("execute")
        is_submit = action.startswith("submit")

        if is_execute or is_submit:
            job_oriented: bool = action.endswith("job")

            if 0 < len(args) <= 2:
                id, time_or_machine = parse_args(args, 2)

                machines = state.tasks[id].machines

                if len(machines) == 1:
                    machine = machines[0]
                    time = time_or_machine

                else:
                    machine = time_or_machine
                    time = -1

            else:
                id, machine, time = parse_args(args, 3)

            instruction = Execute(id, machine, job_oriented, wait=is_submit)

        elif action == "pause":
            task_id, time = parse_args(args, 2)

            instruction = Pause(task_id)

        elif action.startswith("resume"):
            job_oriented = action.endswith("job")

            task_id, time = parse_args(args, 2)

            instruction = Resume(task_id, job_oriented)

        elif action == "complete":
            task_id, time = parse_args(args, 2)

            instruction = Complete(task_id)

        elif action == "advance":
            to_time, time = parse_args(args, 2)

            instruction = Advance(to_time)

        elif action == "noop":
            (time,) = parse_args(args, 1)

            instruction = Advance()

        elif action == "query":
            (time,) = parse_args(args, 1)

            instruction = Query()

        elif action == "clear":
            (time,) = parse_args(args, 1)
            instruction = Clear()

        else:
            raise ValueError(f"Unknown instruction {action}")

    return instruction, time

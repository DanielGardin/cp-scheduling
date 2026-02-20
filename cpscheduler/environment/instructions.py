"""
instructions.py

This module defines the instructions that can be executed in the scheduling environment.
Instructions are used to control the execution of tasks, manage their states, and interact
with the scheduler.
"""

from typing import TypeAlias, Final
from typing_extensions import Self
from collections.abc import Iterable, Iterator
from typing_extensions import Unpack, TypeIs

from enum import Enum

from mypy_extensions import mypyc_attr

from cpscheduler.environment._common import (
    TASK_ID,
    TIME,
    MACHINE_ID,
    Int,
    GLOBAL_MACHINE_ID,
    MAX_TIME,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.utils.list_utils import convert_to_list

SingleAction: TypeAlias = tuple["str | Instruction", Unpack[tuple[Int, ...]]]
ActionType: TypeAlias = SingleAction | Iterable[SingleAction] | None


def is_single_action(
    action: ActionType,
) -> TypeIs[SingleAction]:
    "Check if the action is a single instruction or a iterable of instructions."
    if not isinstance(action, tuple):
        return False

    return isinstance(action[0], str) and all(isinstance(arg, Int) for arg in action[1:])


import logging

logger = logging.getLogger(__name__)

class LogLevel:
    "LogLevel represents the severity of a log message, guiding how it should be handled by the environment."

    CRITICAL: Final[int] = 50
    FATAL: Final[int] = CRITICAL
    ERROR: Final[int] = 40
    WARNING: Final[int] = 30
    WARN: Final[int] = WARNING
    INFO: Final[int] = 20
    DEBUG: Final[int] = 10
    NOTSET: Final[int] = 0


class QueueControl(Enum):
    CONTINUE = 1
    "Continue processing the next instruction in the queue after this one is processed."

    RESTART = 2
    "Restart the queue from the beginning after this instruction is processed."

    BLOCK = 3
    "Block the processing of subsequent instructions in the same time step until this instruction is resolved."

    INTERRUPT = 4
    "Interrupt the processing of subsequent instructions in the same time step and halt."

class InstructionResult:

    done: bool
    "Whether the instruction was successfully processed and should be removed from the schedule."

    queue_control: QueueControl

    log_message: str
    "Optional message to be logged when the instruction is processed."

    level: int
    "The severity level of the log message."

    def __init__(
        self,
        done: bool = False,
        queue_control: QueueControl = QueueControl.CONTINUE,
        log_message: str = "",
        level: int = LogLevel.DEBUG
    ) -> None:
        self.done = done
        self.queue_control = queue_control
        self.log_message = log_message
        self.level = level

    @classmethod
    def success(cls, message: str = "", level: int = LogLevel.DEBUG) -> Self:
        return cls(
            done=True,
            queue_control=QueueControl.CONTINUE,
            log_message=message,
            level=level,
        )

    @classmethod
    def deferred(cls, message: str = "", level: int = LogLevel.DEBUG) -> Self:
        return cls(
            done=False,
            queue_control=QueueControl.CONTINUE,
            log_message=message,
            level=level,
        )

    @classmethod
    def restart(cls, message: str = "", level: int = LogLevel.DEBUG) -> Self:
        return cls(
            done=True,
            queue_control=QueueControl.RESTART,
            log_message=message,
            level=level,
        )

    @classmethod
    def blocked(cls, message: str = "", level: int = LogLevel.DEBUG) -> Self:
        return cls(
            done=False,
            queue_control=QueueControl.BLOCK,
            log_message=message,
            level=level,
        )

    @classmethod
    def halt(cls, message: str = "", level: int = LogLevel.DEBUG) -> Self:
        return cls(
            done=True,
            queue_control=QueueControl.INTERRUPT,
            log_message=message,
            level=level,
        )

    @classmethod
    def invalid(cls, message: str = "", level: int = LogLevel.DEBUG) -> Self:
        return cls(
            done=False,
            queue_control=QueueControl.INTERRUPT,
            log_message=message,
            level=level,
        )
    
SUCCESS = InstructionResult.success()
DEFERRED = InstructionResult.deferred()
RESTART = InstructionResult.restart()
BLOCKED = InstructionResult.blocked()
HALT = InstructionResult.halt()
INVALID = InstructionResult.invalid()

DEFAULT_QUEUE_TIME: Final[TIME] = -1

class Schedule:

    def __init__(self) -> None:
        self.schedule: dict[TIME, list["Instruction"]] = {}
        self.default_queue: list["Instruction"] = list()

    def reset(self) -> None:
        self.schedule.clear()
        self.default_queue.clear()

    def clear_schedule(self) -> None:
        self.schedule.clear()
        self.default_queue.clear()

    def add_instruction(self, instruction: "Instruction", time: TIME = DEFAULT_QUEUE_TIME) -> None:
        if time == DEFAULT_QUEUE_TIME:
            self.default_queue.append(instruction)

        else:
            self.schedule.setdefault(time, []).append(instruction)

    def is_empty(self) -> bool:
        return not (self.schedule or self.default_queue)

    def get_next_instruction_time(self) -> TIME:
        return min(self.schedule) if self.schedule else MAX_TIME

    def instruction_queue(self, state: ScheduleState) -> Iterator[InstructionResult]:
        time = state.time

        if time in self.schedule:
            instructions = self.schedule[time]

            idx =  0
            while idx < len(instructions):
                instruction = instructions[idx]

                result = instruction.apply(state, self)

                if result.log_message:
                    logger.log(result.level, result.log_message)

                if result.done:
                    instructions.pop(idx)

                elif result.queue_control != QueueControl.RESTART:
                    idx += 1

                yield result

                match result.queue_control:
                    case QueueControl.CONTINUE:
                        continue

                    case QueueControl.RESTART:
                        idx = 0

                    case QueueControl.BLOCK:
                        break

                    case QueueControl.INTERRUPT:
                        return

            if instructions:
                # This only means that the queue was blocked and the remaining instructions cannot
                # be processed at the moment, invalidating the schedule for the current time step.
                error_message = (
                    f"Schedule for time {time} is blocked due to instruction {instructions[0]}"
                    f" and cannot process the remaining instructions: {list(instructions)[1:]}"
                )

                logger.error(error_message)

                yield InstructionResult.blocked(error_message, level=LogLevel.ERROR)
                return
            
            self.schedule.pop(time)

        idx = 0
        while idx < len(self.default_queue):
            instruction = self.default_queue[idx]

            result = instruction.apply(state, self)

            if result.log_message:
                logger.log(result.level, result.log_message)

            if result.done:
                self.default_queue.pop(idx)

            elif result.queue_control != QueueControl.RESTART:
                idx += 1

            yield result

            match result.queue_control:
                case QueueControl.CONTINUE:
                    continue

                case QueueControl.RESTART:
                    idx = 0

                case QueueControl.BLOCK:
                    break

                case QueueControl.INTERRUPT:
                    return


def select_machine(task_id: TASK_ID, state: ScheduleState) -> MACHINE_ID:
    "Select a machine for the given task when machine is not specified."
    for machine in state.tasks[task_id].machines:
        if state.is_available(task_id, machine):
            return machine

    return GLOBAL_MACHINE_ID


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

    def is_ready(self, state: ScheduleState) -> bool:
        "Check if the instruction is ready to be processed based on the current state."
        return True

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        "Process the instruction at the given current time."
        raise NotImplementedError

    def lower_bound_time(self, state: ScheduleState) -> TIME:
        "Calculate the lower bound time for the instruction to be ready based on the current state."
        return state.time

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Noop(Instruction):
    "Noop is a no-op instruction that does nothing when applied."

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        return InstructionResult.success()


class Execute(Instruction):
    "Executes a task on a specific machine. If the task cannot be executed, it is waited for."

    def __init__(self, task_id: TASK_ID, machine_id: MACHINE_ID = GLOBAL_MACHINE_ID):
        self.task_id = task_id
        self.machine_id = machine_id

    def __repr__(self) -> str:
        if self.machine_id != GLOBAL_MACHINE_ID:
            return f"Execute(task={self.task_id}, machine={self.machine_id})"

        return f"Execute(task={self.task_id})"

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        if state.is_available(self.task_id, self.machine_id):
            if self.machine_id == GLOBAL_MACHINE_ID:
                self.machine_id = select_machine(self.task_id, state)

            state.execute_task(self.task_id, self.machine_id)

            return SUCCESS

        if state.is_fixed(self.task_id):
            return InstructionResult.invalid(
                f"Task {self.task_id} is already fixed.",
                LogLevel.ERROR,
            )

        if self.machine_id not in state.tasks[self.task_id].machines:
            return InstructionResult.invalid(
                f"Machine {self.machine_id} is not eligible for task {self.task_id}.",
                LogLevel.ERROR,
            )

        return BLOCKED

# @profile
class Submit(Instruction):
    "Submits a task to a specific machine. If the task cannot be executed, it is waited for."

    def __init__(self, id: TASK_ID, machine: MACHINE_ID = GLOBAL_MACHINE_ID):
        self.task_id = id
        self.machine_id = machine

    def __repr__(self) -> str:
        if self.machine_id != GLOBAL_MACHINE_ID:
            return f"Submit(task={self.task_id}, machine={self.machine_id})"

        return f"Submit(task={self.task_id})"

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        if state.is_available(self.task_id, self.machine_id):
            if self.machine_id == GLOBAL_MACHINE_ID:
                self.machine_id = select_machine(self.task_id, state)

            state.execute_task(self.task_id, self.machine_id)

            return SUCCESS

        if self.machine_id not in state.tasks[self.task_id].machines:
            return InstructionResult.invalid(
                f"Machine {self.machine_id} is not eligible for task {self.task_id}.",
                LogLevel.ERROR,
            )

        if state.is_fixed(self.task_id):
            return InstructionResult.invalid(
                f"Task {self.task_id} is already fixed and cannot be submitted.",
                LogLevel.ERROR,
            )

        return DEFERRED


class ExecuteJob(Instruction):
    "Executes all tasks in a job. Can only be used in job-oriented scheduling."

    def __init__(self, job_id: TASK_ID, machine: MACHINE_ID = GLOBAL_MACHINE_ID):
        self.job_id = job_id
        self.machine_id = machine

    def __repr__(self) -> str:
        if self.machine_id != GLOBAL_MACHINE_ID:
            return f"Execute(job={self.job_id}, machine={self.machine_id})"

        return f"Execute(job={self.job_id})"

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        job_tasks = state.jobs[self.job_id]

        for task in job_tasks:
            if state.is_available(task.task_id, self.machine_id):
                machine_id = self.machine_id

                if machine_id == GLOBAL_MACHINE_ID:
                    machine_id = select_machine(task.task_id, state)

                state.execute_task(task.task_id, machine_id)

                return InstructionResult.success(
                    f"Task {task.task_id} in job {self.job_id} executed on machine {machine_id}"
                    f" at {state.time}."
                )

        if all(state.is_fixed(task.task_id) for task in job_tasks):
            return InstructionResult.success(
                f"All tasks in job {self.job_id} are already fixed and cannot be executed."
            )

        return InstructionResult.blocked(
            f"No tasks in job {self.job_id} can be executed on machine {self.machine_id} "
            f"at time {state.time}. Waiting for any of them to become available."
        )


class SubmitJob(Instruction):
    "Submits all tasks in a job. Can only be used in job-oriented scheduling."

    def __init__(self, job_id: TASK_ID, machine: MACHINE_ID = GLOBAL_MACHINE_ID):
        self.job_id = job_id
        self.machine_id = machine

    def __repr__(self) -> str:
        if self.machine_id != GLOBAL_MACHINE_ID:
            return f"Submit(job={self.job_id}, machine={self.machine_id})"

        return f"Submit(job={self.job_id})"

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        job_tasks = state.jobs[self.job_id]

        for task in job_tasks:
            if state.is_available(task.task_id, self.machine_id):
                machine_id = self.machine_id

                if machine_id == GLOBAL_MACHINE_ID:
                    machine_id = select_machine(task.task_id, state)

                state.execute_task(task.task_id, machine_id)

                return InstructionResult.success(
                    f"Task {task.task_id} in job {self.job_id} submitted to machine {machine_id}"
                )

        if all(state.is_fixed(task.task_id) for task in job_tasks):
            return InstructionResult.success(
                f"All tasks in job {self.job_id} are already fixed and cannot be submitted."
            )

        return InstructionResult.deferred(
            f"No tasks in job {self.job_id} can be submitted to machine {self.machine_id} "
            f"at time {state.time}. Skipping them for now until any of them become available."
        )


class Pause(Instruction):
    "Pauses a task if it is currently executing. Can only be used in preemptive scheduling."

    def __init__(self, task_id: TASK_ID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return f"Pause(task={self.task_id})"

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        task = state.tasks[self.task_id]

        if state.is_executing(self.task_id):
            state.pause_task(self.task_id)

            return InstructionResult.success(f"Task {self.task_id} paused at time {state.time}.")

        if not task.preemptive:
            return InstructionResult.success(
                f"Task {self.task_id} is not preemptive and cannot be paused.",
            )

        return InstructionResult.blocked(
            f"Task {self.task_id} cannot be paused at time {state.time}."
        )


class Resume(Instruction):
    "Resumes a paused task in the same machine it was executing before being paused."

    def __init__(self, task_id: TASK_ID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return f"Resume(task={self.task_id})"

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        task = state.tasks[self.task_id]

        if state.is_paused(self.task_id):
            last_machine = state.get_assignment(self.task_id)
            state.execute_task(self.task_id, last_machine)

            return InstructionResult.success(
                f"Task {self.task_id} resumed on machine {last_machine} at time {state.time}."
            )

        if not task.preemptive:
            return InstructionResult.success(
                f"Task {self.task_id} is not preemptive and cannot be resumed."
            )

        if state.is_completed(self.task_id):
            return InstructionResult.success(
                f"Task {self.task_id} is already completed and cannot be resumed."
            )

        return InstructionResult.blocked(
            f"Task {self.task_id} cannot be resumed at time {state.time}.",
        )


class Checkpoint(Instruction):
    "Checkpoint is an no-op instruction used to yield control the default queue, allowing timing instructions."

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        return InstructionResult.success(f"Checkpoint at time {state.time} reached.")


class Complete(Instruction):
    "Advances the current time to the end of an executing task."

    def __init__(self, task_id: TASK_ID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return f"Complete(task={self.task_id})"

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        if state.is_executing(self.task_id):
            end_time = state.get_end_lb(self.task_id)

            schedule.add_instruction(Checkpoint(), end_time)

            return InstructionResult.success(f"Task {self.task_id} completed.")

        if state.is_completed(self.task_id):
            return InstructionResult.success(f"Task {self.task_id} is already completed.")

        return InstructionResult.blocked(
            f"Task {self.task_id} is not executing and cannot be completed at the moment."
        )


class Advance(Instruction):
    "Advances the current time by a specified amount or to the next decision point if not specified."

    def __init__(self, dt: TIME) -> None:
        self.dt = dt

        if self.dt <= 0:
            raise ValueError(f"Advance instruction requires a positive time delta, got {self.dt}.")

    def __repr__(self) -> str:
        return f"Advance(dt={self.dt})"

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        next_time = state.time + self.dt

        schedule.add_instruction(Checkpoint(), next_time)

        return InstructionResult.success(f"Advancing time by {self.dt} to {next_time}.")

class Query(Instruction):
    "When processed, halts the environment and returns its current state."

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        return InstructionResult.halt(f"Querying state at time {state.time}.")


class Clear(Instruction):
    "Clears all upcoming instructions and resets the schedule."

    def apply(self, state: ScheduleState, schedule: Schedule) -> InstructionResult:
        schedule.clear_schedule()

        return InstructionResult.success(f"Clearing schedule at time {state.time}.")


def parse_args(
    args: list[Int],
    output_size: int,
    instruction_name: str = "Instruction",
) -> list[int]:
    "Parse the raw instruction arguments into required and optional arguments."
    if len(args) > output_size:
        raise ValueError(
            f"Too many arguments provided for instruction {instruction_name}. "
            f"Expected at most {output_size}, got {len(args)}."
        )

    return convert_to_list(args, int) + [-1] * (output_size - len(args))


def parse_instruction(
    instruction: str | Instruction, args: list[Int], state: ScheduleState
) -> tuple[Instruction, TIME]:
    "Parse raw instruction arguments into an Instruction object and the scheduled time."

    match instruction:
        case "execute":
            if 0 < len(args) <= 2:
                task_id, time_or_machine = parse_args(args, 2, "execute")

                machines = state.tasks[task_id].machines

                if len(machines) == 1:
                    # Consider the second argument as time
                    machine = next(iter(machines))
                    time = time_or_machine

                else:
                    # Consider the second argument as machine
                    machine = time_or_machine
                    time = -1

            else:
                task_id, machine, time = parse_args(args, 3, "execute")

            return Execute(task_id, machine), time

        case "submit":
            if 0 < len(args) <= 2:
                task_id, time_or_machine = parse_args(args, 2, "submit")

                machines = state.tasks[task_id].machines

                if len(machines) == 1:
                    # Consider the second argument as time
                    machine = next(iter(machines))
                    time = time_or_machine

                else:
                    # Consider the second argument as machine
                    machine = time_or_machine
                    time = -1

            else:
                task_id, machine, time = parse_args(args, 3, "submit")

            return Submit(task_id, machine), time

        case "execute job":
            job_id, machine, time = parse_args(args, 3, "execute job")

            return ExecuteJob(job_id, machine), time

        case "submit job":
            job_id, machine, time = parse_args(args, 3, "submit job")

            return SubmitJob(job_id, machine), time

        case "pause":
            task_id, time = parse_args(args, 2, "pause")

            return Pause(task_id), time

        case "resume":
            task_id, time = parse_args(args, 2, "resume")

            return Resume(task_id), time

        case "complete":
            task_id, time = parse_args(args, 2, "complete")

            return Complete(task_id), time

        case "advance":
            to_time, time = parse_args(args, 2, "advance")

            return Advance(to_time), time

        case "noop":
            (time,) = parse_args(args, 1, "noop")

            return Noop(), time

        case "query":
            (time,) = parse_args(args, 1, "query")

            return Query(), time

        case "clear":
            (time,) = parse_args(args, 1, "clear")

            return Clear(), time

        case Instruction():
            (time,) = parse_args(args, 1, "Instruction")

            return instruction, time

        case _:
            raise ValueError(f"Unknown instruction {instruction}")

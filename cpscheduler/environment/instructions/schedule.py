from typing import TypeAlias, Final
from typing_extensions import Self
from collections.abc import Iterator

from mypy_extensions import mypyc_attr, u8

from cpscheduler.environment.constants import Time, MAX_TIME
from cpscheduler.environment.state import ScheduleState

import logging

logger = logging.getLogger(__name__)

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

QueueControlType: TypeAlias = u8


class QueueControl:
    CONTINUE: Final[QueueControlType] = 1
    "Continue processing the next instruction in the queue after this one is processed."

    RESTART: Final[QueueControlType] = 2
    "Restart the queue from the beginning after this instruction is processed."

    BLOCK: Final[QueueControlType] = 3
    "Block the processing of subsequent instructions in the same time step until this instruction is resolved."

    INTERRUPT: Final[QueueControlType] = 4
    "Interrupt the processing of subsequent instructions in the same time step and halt."


CONTINUE = QueueControl.CONTINUE
RESTART = QueueControl.RESTART
BLOCK = QueueControl.BLOCK
INTERRUPT = QueueControl.INTERRUPT


class InstructionResult:

    done: bool
    "Whether the instruction was successfully processed and should be removed from the schedule."

    queue_control: QueueControlType

    log_message: str
    "Optional message to be logged when the instruction is processed."

    level: int
    "The severity level of the log message."

    def __init__(
        self,
        done: bool = False,
        queue_control: QueueControlType = CONTINUE,
        log_message: str = "",
        level: int = DEBUG,
    ) -> None:
        self.done = done
        self.queue_control = queue_control
        self.log_message = log_message
        self.level = level

    @classmethod
    def success(cls, message: str = "", level: int = DEBUG) -> Self:
        return cls(
            done=True,
            queue_control=CONTINUE,
            log_message=message,
            level=level,
        )

    @classmethod
    def deferred(cls, message: str = "", level: int = DEBUG) -> Self:
        return cls(
            done=False,
            queue_control=CONTINUE,
            log_message=message,
            level=level,
        )

    @classmethod
    def restart(cls, message: str = "", level: int = DEBUG) -> Self:
        return cls(
            done=True,
            queue_control=RESTART,
            log_message=message,
            level=level,
        )

    @classmethod
    def blocked(cls, message: str = "", level: int = DEBUG) -> Self:
        return cls(
            done=False,
            queue_control=BLOCK,
            log_message=message,
            level=level,
        )

    @classmethod
    def halt(cls, message: str = "", level: int = DEBUG) -> Self:
        return cls(
            done=True,
            queue_control=INTERRUPT,
            log_message=message,
            level=level,
        )

    @classmethod
    def invalid(cls, message: str = "", level: int = DEBUG) -> Self:
        return cls(
            done=False,
            queue_control=INTERRUPT,
            log_message=message,
            level=level,
        )


DEFAULT_QUEUE_TIME: Final[Time] = -1


class Schedule:
    schedule: dict[Time, list["Instruction"]]
    default_queue: list["Instruction"]

    def __init__(self) -> None:
        self.schedule = {}
        self.default_queue = []

    def reset(self) -> None:
        self.schedule.clear()
        self.default_queue.clear()

    def clear_schedule(self) -> None:
        self.schedule.clear()
        self.default_queue.clear()

    def add_instruction(
        self, instruction: "Instruction", time: Time = DEFAULT_QUEUE_TIME
    ) -> None:
        if time == DEFAULT_QUEUE_TIME:
            self.default_queue.append(instruction)

        else:
            self.schedule.setdefault(time, []).append(instruction)

    def is_empty(self) -> bool:
        return not (self.schedule or self.default_queue)

    def has_scheduled_instructions(self) -> bool:
        return bool(self.schedule)

    def get_next_instruction_time(self) -> Time:
        return min(self.schedule) if self.schedule else MAX_TIME

    def instruction_queue(
        self, state: ScheduleState
    ) -> Iterator[InstructionResult]:
        time = state.time

        queue = self.schedule.get(time, [])

        idx = 0
        while idx < len(queue):
            instruction = queue[idx]

            result = instruction.apply(state, self)

            control = result.queue_control
            log_message = result.log_message

            if log_message:
                logger.log(result.level, log_message)

            if not result.done:
                if control == CONTINUE:
                    idx += 1

                elif control == RESTART:
                    idx = 0

                elif control == BLOCK:
                    break

                elif control == INTERRUPT:
                    yield result
                    return

            else:
                queue.pop(idx)
                yield result

                if control == RESTART:
                    idx = 0

                elif control == BLOCK:
                    break

                elif control == INTERRUPT:
                    return

        if queue:
            # This only means that the queue was blocked and the remaining instructions cannot
            # be processed at the moment, invalidating the schedule for the current time step.
            error_message = (
                f"Schedule for time {time} is blocked due to instruction {queue[0]}"
                f" and cannot process the remaining instructions: {queue}"
            )

            logger.error(error_message)

            yield InstructionResult.invalid(error_message, level=ERROR)
            return

        self.schedule.pop(time, None)
        queue = self.default_queue

        idx = 0
        while idx < len(queue):
            instruction = queue[idx]

            result = instruction.apply(state, self)

            control = result.queue_control
            log_message = result.log_message

            if not result.done:
                if control == CONTINUE:
                    idx += 1

                elif control == RESTART:
                    idx = 0

                elif control == BLOCK:
                    break

                elif control == INTERRUPT:
                    yield result
                    return

            else:
                queue.pop(idx)
                yield result

                if control == RESTART:
                    idx = 0

                elif control == BLOCK:
                    break

                elif control == INTERRUPT:
                    return


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

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        "Process the instruction at the given current time."
        raise NotImplementedError

    def lower_bound_time(self, state: ScheduleState) -> Time:
        "Calculate the lower bound time for the instruction to be ready based on the current state."
        return state.time

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

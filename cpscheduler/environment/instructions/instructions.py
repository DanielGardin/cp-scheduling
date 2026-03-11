

from cpscheduler.environment.constants import (
    TaskID,
    Time,
    MachineID,
    GLOBAL_MACHINE_ID
)
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.instructions.schedule import (
    Instruction,
    InstructionResult,
    Schedule,
)

import logging
ERROR = logging.ERROR

SUCCESS = InstructionResult.success()
DEFERRED = InstructionResult.deferred()
SUCCESS_RESTART = InstructionResult.restart()
BLOCKED = InstructionResult.blocked()
HALT = InstructionResult.halt()
INVALID = InstructionResult.invalid()

# TODO: Consider adding machine dispatchers (task, state) -> machine_id
def select_machine(task_id: TaskID, state: ScheduleState) -> MachineID:
    "Select a machine for the given task when machine is not specified."
    for machine in state.instance.get_machines(task_id):
        if state.is_available(task_id, machine):
            return machine

    return GLOBAL_MACHINE_ID

class Noop(Instruction):
    "Noop is a no-op instruction that does nothing when applied."

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        return InstructionResult.success()


class Execute(Instruction):
    "Executes a task on a specific machine. If the task cannot be executed, it is waited for."

    def __init__(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ):
        self.task_id = task_id
        self.machine_id = machine_id

    def __repr__(self) -> str:
        if self.machine_id != GLOBAL_MACHINE_ID:
            return f"Execute(task={self.task_id}, machine={self.machine_id})"

        return f"Execute(task={self.task_id})"

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        if state.is_available(self.task_id, self.machine_id):
            if self.machine_id == GLOBAL_MACHINE_ID:
                self.machine_id = select_machine(self.task_id, state)

            state.execute_task(self.task_id, self.machine_id)

            return SUCCESS

        if self.machine_id not in state.instance.get_machines(self.task_id):
            return InstructionResult.invalid(
                f"Machine {self.machine_id} is not eligible for task {self.task_id}.",
                ERROR,
            )

        return BLOCKED


class Submit(Instruction):
    "Submits a task to a specific machine. If the task cannot be executed, it is waited for."

    def __init__(self, id: TaskID, machine: MachineID = GLOBAL_MACHINE_ID):
        self.task_id = id
        self.machine_id = machine

    def __repr__(self) -> str:
        if self.machine_id != GLOBAL_MACHINE_ID:
            return f"Submit(task={self.task_id}, machine={self.machine_id})"

        return f"Submit(task={self.task_id})"

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        if state.is_available(self.task_id, self.machine_id):
            if self.machine_id == GLOBAL_MACHINE_ID:
                self.machine_id = select_machine(self.task_id, state)

            state.execute_task(self.task_id, self.machine_id)

            return SUCCESS

        if self.machine_id not in state.instance.get_machines(self.task_id):
            return InstructionResult.invalid(
                f"Machine {self.machine_id} is not eligible for task {self.task_id}.",
                ERROR,
            )

        return DEFERRED


class ExecuteJob(Instruction):
    "Executes all tasks in a job. Can only be used in job-oriented scheduling."

    def __init__(
        self, job_id: TaskID, machine: MachineID = GLOBAL_MACHINE_ID
    ):
        self.job_id = job_id
        self.machine_id = machine

    def __repr__(self) -> str:
        if self.machine_id != GLOBAL_MACHINE_ID:
            return f"Execute(job={self.job_id}, machine={self.machine_id})"

        return f"Execute(job={self.job_id})"

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        job_tasks = state.instance.job_tasks[self.job_id]

        for task_id in job_tasks:
            if state.is_available(task_id, self.machine_id):
                machine_id = self.machine_id

                if machine_id == GLOBAL_MACHINE_ID:
                    machine_id = select_machine(task_id, state)

                state.execute_task(task_id, machine_id)

                return InstructionResult.success(
                    f"Task {task_id} in job {self.job_id} executed on machine {machine_id}"
                    f" at {state.time}."
                )

        return InstructionResult.blocked(
            f"No tasks in job {self.job_id} can be executed on machine {self.machine_id} "
            f"at time {state.time}. Waiting for any of them to become available."
        )


class SubmitJob(Instruction):
    "Submits all tasks in a job. Can only be used in job-oriented scheduling."

    def __init__(
        self, job_id: TaskID, machine: MachineID = GLOBAL_MACHINE_ID
    ):
        self.job_id = job_id
        self.machine_id = machine

    def __repr__(self) -> str:
        if self.machine_id != GLOBAL_MACHINE_ID:
            return f"Submit(job={self.job_id}, machine={self.machine_id})"

        return f"Submit(job={self.job_id})"

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        job_tasks = state.instance.job_tasks[self.job_id]

        for task_id in job_tasks:
            if state.is_available(task_id, self.machine_id):
                machine_id = self.machine_id

                if machine_id == GLOBAL_MACHINE_ID:
                    machine_id = select_machine(task_id, state)

                state.execute_task(task_id, machine_id)

                return InstructionResult.success(
                    f"Task {task_id} in job {self.job_id} submitted to machine {machine_id}"
                )

        return InstructionResult.deferred(
            f"No tasks in job {self.job_id} can be submitted to machine {self.machine_id} "
            f"at time {state.time}. Skipping them for now until any of them become available."
        )


class Pause(Instruction):
    "Pauses a task if it is currently executing. Can only be used in preemptive scheduling."

    def __init__(self, task_id: TaskID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return f"Pause(task={self.task_id})"

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        if state.is_executing(self.task_id):
            state.pause_task(self.task_id)

            return InstructionResult.success(
                f"Task {self.task_id} paused at time {state.time}."
            )

        if not state.instance.preemptive[self.task_id]:
            return InstructionResult.success(
                f"Task {self.task_id} is not preemptive and cannot be paused.",
            )

        return InstructionResult.blocked(
            f"Task {self.task_id} cannot be paused at time {state.time}."
        )


class Resume(Instruction):
    "Resumes a paused task in the same machine it was executing before being paused."

    def __init__(self, task_id: TaskID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return f"Resume(task={self.task_id})"

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        if state.is_paused(self.task_id):
            last_machine = state.get_assignment(self.task_id)
            state.execute_task(self.task_id, last_machine)

            return InstructionResult.success(
                f"Task {self.task_id} resumed on machine {last_machine} at time {state.time}."
            )

        if not state.instance.preemptive[self.task_id]:
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

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        return InstructionResult.success(
            f"Checkpoint at time {state.time} reached."
        )


class Complete(Instruction):
    "Advances the current time to the end of an executing task."

    def __init__(self, task_id: TaskID):
        self.task_id = task_id

    def __repr__(self) -> str:
        return f"Complete(task={self.task_id})"

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        if state.is_executing(self.task_id):
            end_time = state.get_end_lb(self.task_id)

            schedule.add_instruction(Checkpoint(), end_time)

            return InstructionResult.success(f"Task {self.task_id} completed.")

        if state.is_completed(self.task_id):
            return InstructionResult.success(
                f"Task {self.task_id} is already completed."
            )

        return InstructionResult.blocked(
            f"Task {self.task_id} is not executing and cannot be completed at the moment."
        )


class Advance(Instruction):
    "Advances the current time by a specified amount or to the next decision point if not specified."

    def __init__(self, dt: Time) -> None:
        self.dt = dt

        if self.dt <= 0:
            raise ValueError(
                f"Advance instruction requires a positive time delta, got {self.dt}."
            )

    def __repr__(self) -> str:
        return f"Advance(dt={self.dt})"

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        next_time = state.time + self.dt

        schedule.add_instruction(Checkpoint(), next_time)

        return InstructionResult.success(
            f"Advancing time by {self.dt} to {next_time}."
        )


class Query(Instruction):
    "When processed, halts the environment and returns its current state."

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        return InstructionResult.halt(f"Querying state at time {state.time}.")


class Clear(Instruction):
    "Clears all upcoming instructions and resets the schedule."

    def apply(
        self, state: ScheduleState, schedule: Schedule
    ) -> InstructionResult:
        schedule.clear_schedule()

        return InstructionResult.success(
            f"Clearing schedule at time {state.time}."
        )

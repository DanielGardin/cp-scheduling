"""Instructions for the time-stepped backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import Self, override

from cpscheduler.environment.backend.actions import Instruction
from cpscheduler.environment.backend.step.step import StepBackend
from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MachineID,
    TaskID,
    Time,
)

if TYPE_CHECKING:
    from cpscheduler.environment.state import ScheduleState


def select_machine(
    state: ScheduleState,
    task_id: TaskID,
    time: Time,
) -> MachineID:
    """Select a feasible machine for a task at the current step."""
    machines = state.get_machines(task_id)

    for machine in sorted(machines):
        if state.can_start(task_id, time, machine):
            return machine

    raise ValueError(f"No feasible machine for task {task_id} at time {time}.")


class ExecuteInstruction(Instruction[StepBackend]):
    """Instruction for starting a task at the current time step."""

    task_id: TaskID
    machine_id: MachineID

    def __init__(
        self,
        task_id: TaskID,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> None:
        super().__init__()

        self.task_id = task_id
        self.machine_id = machine_id

    @override
    def resolve(self, state: ScheduleState) -> Self:
        task_id = self.task_id
        if not 0 <= task_id < state.n_tasks:
            raise ValueError(f"Task {task_id} in {self} does not exist.")

        machine_id = self.machine_id
        task_machines = state.get_machines(task_id)
        if machine_id != GLOBAL_MACHINE_ID:
            if machine_id not in task_machines:
                raise ValueError(
                    f"Machine {machine_id} is not available for task {task_id} "
                    f"in {self}"
                )

        elif len(task_machines) == 1:
            resolved_machine = next(iter(task_machines))
            return type(self)(task_id, resolved_machine)

        return self

    def resolve_machine(self, state: ScheduleState, time: Time) -> MachineID:
        """Resolve the machine used to start the task at the current step."""
        if self.machine_id != GLOBAL_MACHINE_ID:
            return self.machine_id

        return select_machine(state, self.task_id, time)

    @override
    def process(self, state: ScheduleState, backend: StepBackend) -> None:
        time = backend.time
        machine_id = self.resolve_machine(state, time)

        if not state.can_start(self.task_id, time, machine_id):
            raise ValueError(
                f"Task {self.task_id} cannot start at time {time} on machine "
                f"{machine_id}."
            )

        state.assign_task(self.task_id, machine_id, time)

    @override
    def semantic(
        self, state: ScheduleState, backend: StepBackend
    ) -> dict[str, Any]:
        machine_id = (
            select_machine(state, self.task_id, backend.time)
            if self.machine_id == GLOBAL_MACHINE_ID
            else self.machine_id
        )

        return {
            "type": "execution",
            "task": self.task_id,
            "machine": machine_id,
        }

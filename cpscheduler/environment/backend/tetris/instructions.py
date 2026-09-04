"""Instructions for the Tetris schedule generation backend."""

from typing import Any

from typing_extensions import Self, override

from cpscheduler.environment.backend.actions import Instruction
from cpscheduler.environment.backend.tetris.tetris import TetrisBackend
from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MachineID,
    TaskID,
)
from cpscheduler.environment.state import ScheduleState


def select_machine(
    state: ScheduleState,
    task_id: TaskID,
) -> MachineID:
    """Select a machine for a task using a simple deterministic heuristic."""
    machines = state.get_machines(task_id)

    for machine in sorted(machines):
        start = state.get_start_lb(task_id, machine)

        if start < state.get_latest_end():
            return machine

    raise ValueError(f"No feasible machine for task {task_id}.")


class ExecuteInstruction(Instruction[TetrisBackend]):
    """Instruction for appending a task to the partial schedule.

    The task is assigned to the specified machine, or to a machine selected
    automatically when ``machine_id`` is ``GLOBAL_MACHINE_ID``. The task is
    then scheduled at its earliest feasible start time.

    Unlike a DES execution event, this instruction does not have an associated
    simulation time. The Tetris schedule generation scheme determines the
    start time from the current partial schedule.
    """

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
            # Statically resolve the machine if there is only one option
            resolved_machine = next(iter(task_machines))
            return type(self)(task_id, resolved_machine)

        return self

    def resolve_machine(self, state: ScheduleState) -> MachineID:
        """Resolve the machine used to append the task."""
        if self.machine_id != GLOBAL_MACHINE_ID:
            return self.machine_id

        return select_machine(state, self.task_id)

    @override
    def process(
        self,
        state: ScheduleState,
        backend: TetrisBackend,
    ) -> None:
        machine_id = self.resolve_machine(state)
        start = state.get_start_lb(self.task_id, machine_id)

        state.assign_task(
            self.task_id,
            machine_id,
            start,
        )

    @override
    def semantic(
        self, state: ScheduleState, backend: TetrisBackend
    ) -> dict[str, Any]:
        machine_id = (
            select_machine(state, self.task_id)
            if self.machine_id == GLOBAL_MACHINE_ID
            else self.machine_id
        )

        return {
            "type": "execution",
            "task": self.task_id,
            "machine": machine_id,
        }

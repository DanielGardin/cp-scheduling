from typing_extensions import Self

from cpscheduler.environment.constants import (
    TaskID,
    MachineID,
    Time,
    GLOBAL_MACHINE_ID,
)
from cpscheduler.environment.des.base import SimulationEvent, Schedule
from cpscheduler.environment.state import ScheduleState


def select_machine(
    state: ScheduleState, task_id: TaskID, machine_id: MachineID
) -> int:
    for machine in state.get_machines(task_id):
        if state.is_available(task_id, machine):
            return machine

    raise ValueError(f"No available machines for task {task_id}.")


class ExecuteEvent(SimulationEvent):
    blocking = True

    __args__ =  ("task_id", "machine_id")

    task_id: TaskID
    machine_id: MachineID

    def __init__(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> None:
        super().__init__()

        self.task_id = task_id
        self.machine_id = machine_id

    def resolve(self, state: ScheduleState) -> Self:
        machine_id = self.machine_id
        task_id = self.task_id
        task_machines = state.get_machines(task_id)

        if machine_id != GLOBAL_MACHINE_ID:
            if machine_id not in task_machines:
                raise ValueError(
                    f"Machine {machine_id} is not available for task {task_id}"
                )

        elif len(task_machines) == 1:
            # Statically resolve the machine if there is only one option
            resolved_machine = next(iter(task_machines))
            return type(self)(task_id, resolved_machine)

        return self

    def earliest_time(self, state: ScheduleState) -> Time:
        return state.get_start_lb(self.task_id, self.machine_id)

    def is_ready(self, state: ScheduleState) -> bool:
        return state.is_available(self.task_id, self.machine_id)

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        machine = self.machine_id
        if machine == GLOBAL_MACHINE_ID:
            machine = select_machine(state, self.task_id, self.machine_id)

        state.execute_task(self.task_id, machine)


class SubmitEvent(ExecuteEvent):
    blocking = False


class PauseEvent(SimulationEvent):
    blocking = True

    __args__ =  ("task_id",)

    task_id: TaskID

    def __init__(self, task_id: int) -> None:
        super().__init__()

        self.task_id = task_id

    def is_ready(self, state: ScheduleState) -> bool:
        return state.is_executing(self.task_id)

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        state.pause_task(self.task_id)


class ResumeEvent(SimulationEvent):
    blocking = True

    __args__ =  ("task_id",)

    task_id: TaskID

    def __init__(self, task_id: int) -> None:
        super().__init__()

        self.task_id = task_id

    @property
    def args(self) -> tuple[int]:
        return (self.task_id,)

    def is_ready(self, state: ScheduleState) -> bool:
        if state.is_paused(self.task_id):
            last_assignment = state.runtime.get_assignment(self.task_id)

            return state.is_available(self.task_id, last_assignment)

        return False

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        last_assignment = state.runtime.get_assignment(self.task_id)
        state.execute_task(self.task_id, last_assignment)


class CheckpointEvent(SimulationEvent):
    blocking = False


class InterruptEvent(SimulationEvent):
    blocking = True


class CompleteEvent(SimulationEvent):
    blocking = True

    __args__ =  ("task_id",)

    task_id: TaskID

    def __init__(self, task_id: int) -> None:
        super().__init__()

        self.task_id = task_id

    def is_ready(self, state: ScheduleState) -> bool:
        return state.is_executing(self.task_id)

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        schedule.add_event(
            CheckpointEvent(), state, state.get_end_lb(self.task_id)
        )


class AdvanceTimeEvent(SimulationEvent):
    blocking = True

    __args__ =  ("dt",)

    dt: Time

    def __init__(self, dt: Time) -> None:
        super().__init__()

        self.dt = dt

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        # Advance time by adding a no-op event at the target time
        schedule.add_event(CheckpointEvent(), state, state.time + self.dt)

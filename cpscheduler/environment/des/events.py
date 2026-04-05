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
    task_machines = state.instance.get_machines(task_id)
    for machine in task_machines:
        if state.is_available(task_id, machine):
            return machine

    raise ValueError(f"No available machines for task {task_id}.")


class ExecuteEvent(SimulationEvent):
    blocking = True

    def __init__(
        self, task_id: TaskID, machine_id: MachineID = GLOBAL_MACHINE_ID
    ) -> None:
        self.task_id = task_id
        self.machine_id = machine_id

    @property
    def args(self) -> tuple[int, int]:
        return (self.task_id, self.machine_id)

    def resolve(self, state: ScheduleState) -> None:
        task_machines = state.instance.get_machines(self.task_id)

        if self.machine_id != GLOBAL_MACHINE_ID:
            if self.machine_id not in state.instance.get_machines(self.task_id):
                raise ValueError(
                    f"Machine {self.machine_id} is not available for task {self.task_id}"
                )

        elif len(task_machines) == 1:
            # Statically resolve the machine if there is only one option
            self.machine_id = next(iter(task_machines))

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

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id

    @property
    def args(self) -> tuple[int]:
        return (self.task_id,)

    def is_ready(self, state: ScheduleState) -> bool:
        return state.is_executing(self.task_id)

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        state.pause_task(self.task_id)


class ResumeEvent(SimulationEvent):
    blocking = True

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id

    @property
    def args(self) -> tuple[int]:
        return (self.task_id,)

    def is_ready(self, state: ScheduleState) -> bool:
        if state.is_paused(self.task_id):
            last_assignment = state.runtime_state.get_assignment(self.task_id)

            return state.is_available(self.task_id, last_assignment)

        return False

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        last_assignment = state.runtime_state.get_assignment(self.task_id)
        state.execute_task(self.task_id, last_assignment)


class CheckpointEvent(SimulationEvent):
    blocking = False

    @property
    def args(self) -> tuple[()]:
        return ()


class InterruptEvent(SimulationEvent):
    blocking = True

    @property
    def args(self) -> tuple[()]:
        return ()


class CompleteEvent(SimulationEvent):
    blocking = True

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id

    @property
    def args(self) -> tuple[int]:
        return (self.task_id,)

    def is_ready(self, state: ScheduleState) -> bool:
        return state.is_executing(self.task_id)

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        schedule.add_event(
            CheckpointEvent(), state, state.get_end_lb(self.task_id)
        )


class AdvanceTimeEvent(SimulationEvent):
    blocking = True

    def __init__(self, dt: Time) -> None:
        self.time = dt

    @property
    def args(self) -> tuple[int]:
        return (self.time,)

    def process(self, state: ScheduleState, schedule: Schedule) -> None:
        # Advance time by adding a no-op event at the target time
        schedule.add_event(CheckpointEvent(), state, state.time + self.time)

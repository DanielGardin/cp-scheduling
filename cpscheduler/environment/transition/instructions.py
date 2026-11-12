"""Built-in instructions, the actions that every transition understands."""

from typing import Any

from typing_extensions import Self, override

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.transition.actions import Instruction
from cpscheduler.environment.transition.core import (
    Clock,
    GlobalClock,
    Transition,
)


class ExecuteInstruction(Instruction[Clock]):
    """Instruction for placing a task in the partial schedule.

    The placement of the transition gives the task its machine, unless one is
    selected, and its start. The task must belong to the candidate set.
    """

    blocking = True

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

        if state.is_fixed(task_id):
            raise ValueError(f"Task {task_id} in {self} is already fixed.")

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
            return type(self)(task_id, task_machines[0])

        return self

    @override
    def earliest_time(self, state: ScheduleState) -> Time:
        return state.get_start_lb(self.task_id, self.machine_id)

    @override
    def is_ready(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> bool:
        return transition.candidate.is_candidate(
            state, transition.clock, self.task_id, self.machine_id
        )

    @override
    def is_locked(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> bool:
        return state.is_locked(self.task_id)

    @override
    def process(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> None:
        machine_id, start = transition.place(
            state, self.task_id, self.machine_id
        )

        transition.placement.apply(state, self.task_id, machine_id, start)

    @override
    def semantic(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> dict[str, Any]:
        machine_id = self.machine_id

        # Before the task is a candidate, its machine is not known yet.
        if machine_id == GLOBAL_MACHINE_ID and self.is_ready(state, transition):
            machine_id, _ = transition.place(state, self.task_id, machine_id)

        return {
            "type": "execution",
            "task": self.task_id,
            "machine": machine_id,
        }


class SubmitInstruction(ExecuteInstruction):
    """Instruction for placing a task as soon as it is a candidate.

    Unlike `execute`, it does not hold the queue: the instructions queued
    after it are processed while it waits. Submitting every task in some
    order decodes a priority list, the parallel schedule generation scheme
    when the clock advances.
    """

    blocking = False


class RejectInstruction(Instruction[Clock]):
    """Instruction for leaving an optional task out of the schedule.

    Rejecting a required task makes the state infeasible.
    """

    task_id: TaskID

    def __init__(self, task_id: TaskID) -> None:
        super().__init__()

        self.task_id = task_id

    @override
    def resolve(self, state: ScheduleState) -> Self:
        if not 0 <= self.task_id < state.n_tasks:
            raise ValueError(f"Task {self.task_id} in {self} does not exist.")

        return self

    @override
    def process(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> None:
        state.forbid_task(self.task_id)


class HaltInstruction(Instruction[Clock]):
    """Instruction for ending the current step when it is processed.

    The instructions queued after it wait for the next step.
    """

    blocking = True

    @override
    def process(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> None:
        transition.clock.halt()


class SkipInstruction(Instruction[Clock]):
    """Instruction that holds the queue and does nothing when processed."""

    blocking = True


class NoopInstruction(Instruction[GlobalClock]):
    """Instruction for waiting until the next time a task can start.

    The untimed instructions queued after it wait too. It needs a clock with
    time, as waiting has no meaning without one.
    """

    blocking = True

    @override
    def process(
        self, state: ScheduleState, transition: Transition[GlobalClock]
    ) -> None:
        clock = transition.clock
        next_time = state.get_next_decision_point(clock.time)

        if next_time == MAX_TIME:
            return

        clock.advance_to(next_time)


class CompleteInstruction(Instruction[GlobalClock]):
    """Instruction for waiting until a task completes.

    It is ready once the task is placed, and the untimed instructions queued
    after it wait until the end of the task.
    """

    blocking = True

    task_id: TaskID

    def __init__(self, task_id: TaskID) -> None:
        super().__init__()

        self.task_id = task_id

    @override
    def earliest_time(self, state: ScheduleState) -> Time | None:
        return state.get_start_lb(self.task_id)

    @override
    def is_ready(
        self, state: ScheduleState, transition: Transition[GlobalClock]
    ) -> bool:
        return state.is_fixed(self.task_id) and state.is_present(self.task_id)

    @override
    def is_locked(
        self, state: ScheduleState, transition: Transition[GlobalClock]
    ) -> bool:
        # Only an instruction placing the task can make it ready.
        return not state.is_fixed(self.task_id)

    @override
    def process(
        self, state: ScheduleState, transition: Transition[GlobalClock]
    ) -> None:
        transition.clock.advance_to(state.get_end(self.task_id))


class FinishEpisodeInstruction(Instruction[Clock]):
    """Instruction for ending the episode when it is processed.

    Any instructions queued after it are ignored, and any remaining tasks are
    rejected.
    """

    blocking = True

    @override
    def process(
        self, state: ScheduleState, transition: Transition[Clock]
    ) -> None:
        for task_id in state.get_unassigned_tasks():
            state.forbid_task(task_id)

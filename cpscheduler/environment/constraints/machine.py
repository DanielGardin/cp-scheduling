from typing import Any
from typing_extensions import Self
from collections.abc import Iterable, Mapping

from cpscheduler.environment.constants import (
    TaskID,
    MachineID,
    Time,
    Int,
    GLOBAL_MACHINE_ID,
    MAX_TIME,
)

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils import convert_to_list

from cpscheduler.environment.constraints.base import Constraint


class MachineEligibilityConstraint(Constraint):
    """
    Machine eligibility constraint for the scheduling environment.
    This constraint defines the machines on which each task can be executed.

    Arguments:
        eligibility: Mapping[int, Iterable[int]]
            A mapping of task IDs to a list of machine IDs on which the task can be executed.

    Note:
        This constraint is limited by the setup of the scheduling environment,
        meaning that you cannot:
        - add machines that do not exist in the environment.
        - include/exclude machines that would make the task incompatible with the scheduling setup.
        - exclude all machines for a task.

        By default, if eligibility is not defined for a task, it is assumed that the task
        can be executed on the original set of machines defined by the scheduling setup.
    """

    eligibility: dict[TaskID, set[MachineID]]

    def __init__(self, eligibility: Mapping[Int, Iterable[Int]]):
        self.eligibility = {
            TaskID(task): {MachineID(machine) for machine in machines}
            for task, machines in eligibility.items()
        }

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.eligibility,),
            (),
        )

    def reset(self, state: ScheduleState) -> None:
        for task_id, machines in self.eligibility.items():
            for other_machine in state.get_machines(task_id):
                if other_machine not in machines:
                    state.forbid_machine(task_id, other_machine)

    def on_bound_reset(self, task_id: TaskID, state: ScheduleState) -> None:
        machines = self.eligibility[task_id]

        for other_machine in state.get_machines(task_id):
            if other_machine not in machines:
                state.forbid_machine(task_id, other_machine)

    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        return self.on_bound_reset(task_id, state)

    def get_entry(self) -> str:
        return "M_j"


class MachineConstraint(Constraint):
    """
    General Parallel machine constraint, differs from NonOverlapConstraint as the disjunctive
    constraint have groups with predefined tasks, while the machine constraint defines its groups
    based on the machine assignment of the tasks.

    This is mainly a setup constraint, if the expected behavior is to constrain tasks on which
    machines they can be assigned to, use the MachineEligibilityConstraint instead.
    """

    machine_map: list[set[TaskID]]

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (),
            (self.machine_map,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.machine_map,) = state

    def reset(self, state: ScheduleState) -> None:
        self.machine_map = [set() for _ in range(state.n_machines)]

        for task_id, machines in enumerate(state.instance.processing_times):
            for machine in machines:
                self.machine_map[machine].add(TaskID(task_id))

    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if machine_id == GLOBAL_MACHINE_ID:
            for machine in state.instance.get_machines(task_id):
                self.machine_map[machine].discard(task_id)

        else:
            self.machine_map[machine_id].discard(task_id)

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for machine in state.instance.get_machines(task_id):
            self.machine_map[machine].discard(task_id)

        end_time = state.get_end_lb(task_id)

        for other_task_id in self.machine_map[machine_id]:
            state.tight_start_lb(other_task_id, end_time, machine_id)

    def on_pause(self, task_id: TaskID, machine_id: MachineID, state: ScheduleState) -> None:
        for other_task_id in self.machine_map[machine_id]:
            state.reset_bounds(other_task_id)

        for machine in state.instance.get_machines(task_id):
            self.machine_map[machine].add(task_id)

class MachineBreakdownConstraint(Constraint):
    """
    Machine breakdown constraint for the scheduling environment.

    This constraint defines the breakdowns for machines, which are the times
    when the machines are unavailable for task execution. The breakdowns can be defined
    as a mapping of machine IDs to a list of (start_time, end_time) tuples representing
    the breakdown intervals.

    Arguments:
        breakdowns: Mapping[int, Iterable[tuple[int, int]]]
            A mapping of machine IDs to a list of (start_time, end_time) tuples
            representing the breakdown intervals for each machine.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    breakdowns: dict[MachineID, list[tuple[Time, Time]]]
    next_breakdown: dict[MachineID, int]

    def __init__(self, breakdowns: Mapping[Int, Iterable[tuple[Int, Int]]]):
        self.breakdowns = {
            MachineID(machine): sorted(
                (Time(start), Time(end)) for start, end in intervals
            )
            for machine, intervals in breakdowns.items()
        }

        self.next_breakdown = {machine: 0 for machine in self.breakdowns}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.breakdowns,),
            (self.next_breakdown,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.next_breakdown,) = state

    @classmethod
    def from_machine_step_function(
        cls, step_function: Mapping[Int, Int]
    ) -> Self:
        """
        Create a MachineBreakdownConstraint from a machine step function.
        This function is only useful in Parallel Machine Environments, where machines are
        indistinguishable and if suffices to define the number of available machines at each
        time step.

        Arguments:
            step_function: Mapping[int, int]
                A mapping of time steps to the number of available machines from that time onward.

            name: Optional[str] = None
                An optional name for the constraint.
        """
        breakdowns: dict[Int, list[tuple[Int, Int]]] = {}

        sorted_times = sorted([int(time) for time in step_function.keys()])
        n_machines = max(int(count) for count in step_function.values())

        for time in sorted_times:
            available_machines = int(step_function[Time(time)])

            for machine in range(available_machines):
                if machine not in breakdowns:
                    continue

                start, end = breakdowns[machine][-1]
                if end == MAX_TIME:
                    breakdowns[machine][-1] = (start, time)

            for machine in range(available_machines, n_machines):
                if (
                    breakdowns.setdefault(machine, [])
                    and breakdowns[machine][-1][1] == time
                ):
                    continue

                breakdowns[machine].append((time, MAX_TIME))

        return cls(breakdowns)

    def reset(self, state: ScheduleState) -> None:
        for machine in self.breakdowns:
            self.next_breakdown[machine] = 0

            for task_id in list(state.runtime_state.awaiting_tasks):
                start_lb = state.get_start_lb(task_id, machine)

                for _, end in self.breakdowns[machine]:
                    if start_lb < end:
                        state.tight_start_lb(task_id, end, machine)

                    else:
                        self.next_breakdown[machine] += 1

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        for machine, breakdown_intervals in self.breakdowns.items():
            current_index = self.next_breakdown[machine]

            while current_index < len(breakdown_intervals):
                start, end = breakdown_intervals[current_index]

                if end <= time:
                    self.next_breakdown[machine] += 1
                    current_index += 1
                    continue

                if start <= time < end:
                    for task_id in list(state.runtime_state.awaiting_tasks):
                        start_lb = state.get_start_lb(task_id, machine)

                        if start_lb < end:
                            state.tight_start_lb(task_id, end, machine)

                break

    def get_entry(self) -> str:
        return "brkdwn"


class BatchConstraint(Constraint):
    """
    Parallel batch machine constraint with capacity b.

    Generalization of MachineConstraint (which corresponds to b = 1).
    Each machine can process up to `capacity[m]` tasks simultaneously,
    """
    constant_capacity: int | None

    machine_map: list[set[TaskID]]

    capacity: list[int]
    running_tasks: list[int]
    next_free_time: list[Time]

    def __init__(self, capacity: Iterable[Int] | Int):
        super().__init__()

        if isinstance(capacity, int):
            if capacity < 1:
                raise ValueError(f"Cannot set capacity any lower than 1.")

            self.constant_capacity = capacity
            self.capacity = []

        else:
            self.constant_capacity = None
            self.capacity = convert_to_list(capacity, int)

        self.running_tasks = []
        self.next_free_time = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.capacity,),
            (self.machine_map, self.running_tasks, self.next_free_time),
        )

    def initialize(self, state: ScheduleState) -> None:
        if self.constant_capacity is not None:
            self.capacity = [self.constant_capacity] * state.n_machines

    def reset(self, state: ScheduleState) -> None:
        self.machine_map = [set() for _ in range(state.n_machines)]

        for task_id in range(state.n_tasks):
            for machine in state.get_machines(task_id):
                self.machine_map[machine].add(TaskID(task_id))

        n_machines = state.n_machines

        self.running_tasks = [0] * n_machines
        self.next_free_time = [0] * n_machines

    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if machine_id == GLOBAL_MACHINE_ID:
            for machine in state.instance.get_machines(task_id):
                self.machine_map[machine].discard(task_id)

        else:
            self.machine_map[machine_id].discard(task_id)

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for machine in state.instance.get_machines(task_id):
            self.machine_map[machine].discard(task_id)

        end_time = state.get_end_lb(task_id)

        if end_time > self.next_free_time[machine_id]:
            self.next_free_time[machine_id] = end_time

        self.running_tasks[machine_id] += 1

        if self.running_tasks[machine_id] == self.capacity[machine_id]:
            for other_task_id in self.machine_map[machine_id]:
                state.tight_start_lb(other_task_id, end_time, machine_id)

    def on_pause(self, task_id: TaskID, machine_id: MachineID, state: ScheduleState) -> None:
        raise NotImplementedError()

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        for machine_id, free_time in enumerate(self.next_free_time):
            if free_time < time:
                continue

            if free_time == time:
                self.running_tasks[machine_id] = 0
                continue

            if self.running_tasks[machine_id] < self.capacity[machine_id]:
                for other_task_id in self.machine_map[machine_id]:
                    state.tight_start_lb(other_task_id, free_time, machine_id)

from collections.abc import Iterable, Mapping

from typing_extensions import Self

from cpscheduler.environment.constants import (
    MAX_TIME,
    Int,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.instance import (
    Feature,
    MachineFeature,
    ProblemInstance,
)
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list, extend_list


class MachineConstraint(Constraint):
    """
    General Parallel machine constraint, differs from NonOverlapConstraint as the disjunctive
    constraint have groups with predefined tasks, while the machine constraint defines its groups
    based on the machine assignment of the tasks.

    This is mainly a setup constraint, if the expected behavior is to constrain tasks on which
    machines they can be assigned to, use the MachineEligibilityConstraint instead.
    """

    machine_map: list[set[TaskID]]

    def initialize(self, instance: ProblemInstance) -> None:
        self.machine_map = [set() for _ in range(instance.n_machines)]

    def reset(self, state: ScheduleState) -> None:
        for tasks in self.machine_map:
            tasks.clear()

        for task_id in range(state.n_tasks):
            for machine in state.get_machines(task_id):
                self.machine_map[machine].add(TaskID(task_id))

    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.machine_map[machine_id].discard(task_id)

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for machine in state.get_original_machines(task_id):
            self.machine_map[machine].discard(task_id)

        end_time = state.get_end_lb(task_id)

        for other_task_id in self.machine_map[machine_id]:
            state.tight_start_lb(other_task_id, end_time, machine_id)

    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for other_task_id in self.machine_map[machine_id]:
            state.reset_bounds(other_task_id)

        for machine in state.get_original_machines(task_id):
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

    breakdowns: MachineFeature[list[tuple[Time, Time]]]
    next_breakdown: dict[MachineID, int]

    def __init__(
        self,
        breakdowns: Mapping[Int, Iterable[tuple[Int, Int]]] | None = None,
        name: str = "breakdown",
    ):
        self.breakdowns = MachineFeature(
            name=name,
            semantic="calendar",
        )

        if breakdowns is not None:
            n_machines = max(convert_to_list(breakdowns.keys(), int)) + 1

            calendar = [
                [
                    (Time(start), Time(end))
                    for start, end in breakdowns.get(i, [])
                ]
                for i in range(n_machines)
            ]

            self.breakdowns.set_data(calendar)

    def add_breakdown(
        self, machine_id: Int, start_time: Int, end_time: Int
    ) -> None:
        if not self.breakdowns.loaded:
            self.breakdowns.set_data([])

        machine = MachineID(machine_id)
        extend_list(self.breakdowns.value, machine + 1, list)

        self.breakdowns.value[machine].append(
            (Time(start_time), Time(end_time))
        )

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

        sorted_times = sorted([int(time) for time in step_function])
        n_machines = max(int(count) for count in step_function.values())

        for time in sorted_times:
            available_machines = int(step_function[time])

            for machine in range(available_machines):
                if machine not in breakdowns:
                    continue

                start, end = breakdowns[machine][-1]
                if Time(end) == MAX_TIME:
                    breakdowns[machine][-1] = (start, time)

            for machine in range(available_machines, n_machines):
                if (
                    breakdowns.setdefault(machine, [])
                    and breakdowns[machine][-1][1] == time
                ):
                    continue

                breakdowns[machine].append((time, MAX_TIME))

        return cls(breakdowns)

    def get_features(self) -> list[Feature]:
        return [self.breakdowns]

    def initialize(self, instance: ProblemInstance) -> None:
        self.next_breakdown = {
            machine: 0
            for machine, breakdowns in enumerate(self.breakdowns.value)
            if breakdowns
        }

        for machine in self.next_breakdown:
            self.breakdowns.value[machine].sort()

    def reset(self, state: ScheduleState) -> None:
        for machine in self.next_breakdown:
            self.next_breakdown[machine] = 0

        self.on_time_update(state.time, state)

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        breakdowns = self.breakdowns.value

        for machine, current_index in self.next_breakdown.items():
            breakdown_intervals = breakdowns[machine]

            while current_index < len(breakdown_intervals):
                start, end = breakdown_intervals[current_index]

                if end <= time:
                    self.next_breakdown[machine] += 1
                    current_index += 1
                    continue

                if start <= time < end:
                    for task_id in state.runtime.get_awaiting_tasks():
                        start_lb = state.get_start_lb(task_id, machine)

                        if start_lb < end:
                            state.tight_start_lb(task_id, end, machine)

                break

    @classmethod
    def get_general_entry(cls) -> str:
        return "brkdwn"


class BatchConstraint(Constraint):
    """
    Parallel batch machine constraint with capacity b.

    Generalization of MachineConstraint (which corresponds to b = 1).
    Each machine can process up to `capacity[m]` tasks simultaneously,
    """

    capacity: MachineFeature[int]
    constant_capacity: int | None

    machine_map: list[set[TaskID]]
    running_tasks: list[set[TaskID]]
    next_free_time: list[Time]

    def __init__(
        self,
        capacity: Iterable[Int] | Int | None = None,
        name: str = "batch_capacity",
    ):
        self.constant_capacity = None

        if capacity is None:
            self.capacity = MachineFeature(name=name, semantic="count")

        else:
            if isinstance(capacity, Int):
                self.constant_capacity = int(capacity)
                storage: list[int] = []

            else:
                storage = convert_to_list(capacity, int)

            self.capacity = MachineFeature(
                name=name, semantic="count", default=storage
            )

    def set_capacity(self, machine_id: Int, capacity: Int) -> None:
        machine = MachineID(machine_id)

        if self.constant_capacity is not None:
            raise ValueError(
                "Cannot add capacity to a batch constraint with constant capacity."
            )

        extend_list(self.capacity.value, machine + 1, lambda: 1)

        self.capacity.value[machine] = int(capacity)

    def get_features(self) -> list[Feature]:
        return [self.capacity]

    def initialize(self, instance: ProblemInstance) -> None:
        n_machines = instance.n_machines
        cap = self.constant_capacity

        if cap is not None:
            extend_list(self.capacity.value, n_machines, lambda: cap)

        elif len(self.capacity.value) != n_machines:
            raise ValueError(
                f"Capacity list length {len(self.capacity.value)} does not match the number of machines {instance.n_machines}."
            )

    def reset(self, state: ScheduleState) -> None:
        for machine_id, capacity in enumerate(self.capacity.value):
            if capacity <= 0:
                for task_id in state.runtime.get_awaiting_tasks():
                    if machine_id in state.get_machines(task_id):
                        state.forbid_machine(task_id, machine_id)

        n_tasks = state.n_tasks
        n_machines = state.n_machines

        self.machine_map = [set() for _ in range(n_machines)]

        for machine, capacity in enumerate(self.capacity.value):
            if capacity <= 0:
                for task_id in range(n_tasks):
                    state.forbid_machine(task_id, machine)

        for task_id in range(n_tasks):
            for machine in state.get_machines(task_id):
                self.machine_map[machine].add(TaskID(task_id))

        self.running_tasks = [set() for _ in range(n_machines)]
        self.next_free_time = [0] * n_machines

    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.machine_map[machine_id].discard(task_id)

    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for machine in state.get_original_machines(task_id):
            self.machine_map[machine].discard(task_id)

        end_time = state.get_end_lb(task_id)

        if end_time > self.next_free_time[machine_id]:
            self.next_free_time[machine_id] = end_time

        self.running_tasks[machine_id].add(task_id)

        if (
            len(self.running_tasks[machine_id])
            == self.capacity.value[machine_id]
        ):
            for other_task_id in self.machine_map[machine_id]:
                state.tight_start_lb(other_task_id, end_time, machine_id)

    def on_pause(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        # Decrement the running task count on the machine
        self.running_tasks[machine_id].discard(task_id)

        if self.running_tasks[machine_id]:
            self.next_free_time[machine_id] = max(
                state.get_end_lb(running_task_id)
                for running_task_id in self.running_tasks[machine_id]
            )

        else:
            self.next_free_time[machine_id] = state.time

            for other_task_id in self.machine_map[machine_id]:
                state.reset_bounds(other_task_id)

        for machine in state.get_original_machines(task_id):
            self.machine_map[machine].add(task_id)

    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        for machine_id, free_time in enumerate(self.next_free_time):
            if free_time < time:
                continue

            if free_time == time:
                self.running_tasks[machine_id].clear()
                continue

            if (
                len(self.running_tasks[machine_id])
                < self.capacity.value[machine_id]
            ):
                for other_task_id in self.machine_map[machine_id]:
                    state.tight_start_lb(other_task_id, free_time, machine_id)

    def get_entry(self) -> str:
        if self.constant_capacity:
            return f"batch={self.constant_capacity}"

        return "batch"

    @classmethod
    def get_general_entry(cls) -> str:
        return "batch"

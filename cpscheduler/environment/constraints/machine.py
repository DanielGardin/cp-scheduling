"""Machine-related constraints for the scheduling environment."""

from collections.abc import Iterable, Mapping

from typing_extensions import Self, override

from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    Int,
    MachineID,
    TaskID,
    Time,
)
from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.instance import Feature, ProblemInstance
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list, extend_list


# FUTURE: This implementation inherently forces the append semantics in the
# transition, not supporting insertion is an artifact of this implementation, not
# a transition-specific behavior.
# My proposal is to change the behavior of this function to allow gap insertion
# by default, and document that append mechanism is enabled by the
# on_time_update handler, instead of being on by default.
class MachineConstraint(Constraint):
    """General Parallel machine constraint.

    Machine Constraint is a specialized disjunctive constraint that enforces that
    only a single task is processed in a machine at a time.

    This is mainly a setup constraint, if the expected behavior is to constrain
    tasks on which machines they can be assigned to, use the
    `MachineEligibilityConstraint` instead.
    """

    backtrack_safe = True

    machine_map: list[set[TaskID]]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        self.machine_map = [set() for _ in range(instance.n_machines)]

    @override
    def reset(self, state: ScheduleState) -> None:
        for tasks in self.machine_map:
            tasks.clear()

        for task_id in range(state.n_tasks):
            for machine in state.get_machines(task_id):
                self.machine_map[machine].add(TaskID(task_id))

    @override
    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.machine_map[machine_id].discard(task_id)

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for machine in state.get_original_machines(task_id):
            self.machine_map[machine].discard(task_id)

        end_time = state.get_end_lb(task_id)

        for other_task_id in self.machine_map[machine_id]:
            state.tight_start_lb(other_task_id, end_time, machine_id)

    @override
    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        for task_id in changed_tasks:
            if state.is_assigned(task_id):
                continue

            for machine in state.get_machines(task_id):
                self.machine_map[machine].add(TaskID(task_id))


class MachineBreakdownConstraint(Constraint):
    """Machine breakdown constraint for the scheduling environment.

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

    backtrack_safe = True
    time_aware = True

    breakdowns: Feature[list[list[tuple[Time, Time]]]]

    def __init__(
        self,
        breakdowns: Mapping[Int, Iterable[tuple[Int, Int]]] | None = None,
        name: str = "breakdown",
    ):
        """Initialize the Machine Breakdown Constraint.

        Parameters
        ----------
        breakdowns: Mapping[int, Iterable[tuple[int, int]]], optional
            A mapping of machine IDs to a list of (start_time, end_time) tuples
            representing the breakdown intervals for each machine.

        name: str, optional
            An optional name for the breakdown feature.

        """
        self.breakdowns = Feature(
            name=name,
            # Machines without breakdowns may be omitted, so its length
            # cannot be used to infer n_machines.
            shape=(None, None, 2),
            value_type="time",
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

            self.breakdowns.own_data(calendar)

    def clear_breakdowns(self) -> None:
        """Empty all breakdowns in the constraint."""
        self.breakdowns.own_data([])

    def add_breakdown(
        self, machine_id: Int, start_time: Int, end_time: Int
    ) -> None:
        """Add a breakdown interval for a specific machine."""
        data = self.breakdowns.value if self.breakdowns.loaded else []

        machine = MachineID(machine_id)
        extend_list(data, machine + 1, list)
        data[machine].append((Time(start_time), Time(end_time)))

        self.breakdowns.own_data(data)

    @classmethod
    def from_machine_step_function(
        cls, step_function: Mapping[Int, Int]
    ) -> Self:
        """Create a Machine Breakdown Constraint from a step function.

        This function is only well-defined in Identical Parallel Machine Setups,
        where machines are indistinguishable and if suffices to only define the
        number of available machines at each time step.

        Parameters
        ----------
        step_function: Mapping[int, int]
            A mapping of time steps to the number of available machines from
            that time onward.

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

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for intervals in self.breakdowns.value:
            intervals.sort()

    def _first_available_start(
        self, machine_id: MachineID, start: Time, duration: Time
    ) -> Time:
        """Return the first start >= `start` that overlaps no breakdown.

        Tasks are non-preemptive, so they must fit entirely between breakdowns.
        """
        calendar = self.breakdowns.value

        if machine_id >= len(calendar):
            return start

        for break_start, break_end in calendar[machine_id]:
            if start < break_end and break_start < start + duration:
                start = break_end

        return start

    def _push_task(
        self,
        task_id: TaskID,
        machine_id: MachineID,
        time: Time,
        state: ScheduleState,
    ) -> None:
        start = max(time, state.get_start_lb(task_id, machine_id))
        duration = state.get_remaining_time(task_id, machine_id)
        available = self._first_available_start(machine_id, start, duration)

        if available == start:
            return

        if available >= MAX_TIME - duration:
            state.forbid_machine(task_id, machine_id)

        else:
            state.tight_start_lb(task_id, available, machine_id)

    @override
    def reset(self, state: ScheduleState) -> None:
        for task_id in range(state.n_tasks):
            for machine_id in state.get_machines(task_id).copy():
                self._push_task(task_id, machine_id, 0, state)

    def _on_lb_change(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        if state.is_fixed(task_id):
            return

        machines = (
            state.get_machines(task_id).copy()
            if machine_id == GLOBAL_MACHINE_ID
            else [machine_id]
        )

        for machine in machines:
            if state.is_feasible(task_id, machine):
                self._push_task(task_id, machine, 0, state)

    @override
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._on_lb_change(task_id, machine_id, state)

    @override
    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self._on_lb_change(task_id, machine_id, state)

    @override
    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        for task_id in state.get_unassigned_tasks():
            for machine_id in state.get_machines(task_id).copy():
                self._push_task(task_id, machine_id, time, state)

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        # Safeguard for clocks that jump without notice (e.g. a timed instruction
        # that was already validated): a schedule must never cross a breakdown.
        start = state.get_start(task_id)
        duration = state.get_remaining_time(task_id, machine_id)

        if self._first_available_start(machine_id, start, duration) != start:
            state.fail(task_id)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "brkdwn"


class BatchConstraint(Constraint):
    """Parallel batch machine constraint with capacity b.

    Generalization of MachineConstraint (which corresponds to b = 1).
    Each machine can process up to `capacity[m]` tasks simultaneously.
    """

    capacity: Feature[list[int]]
    constant_capacity: int | None

    machine_map: list[set[TaskID]]

    _release_times: list[list[Time]]
    """For each machine: [MAX_TIME, t_1, ..., t_k], with t_1 >= ... >= t_k the
    times at which a running task releases a slot."""

    _free_slots: list[list[int]]
    """For each machine: free slots during each interval [t_j, t_{j-1})."""

    def __init__(
        self,
        capacity: Iterable[Int] | Int | None = None,
        name: str = "batch_capacity",
    ):
        """Initialize the Batch Constraint.

        Parameters
        ----------
        capacity: Iterable[int] | int | None, optional
            The capacity for each machine.
            If an integer is provided, it is used as a constant capacity for all machines.
            If None is provided, the capacity is expected to be set later.

        name: str, optional
            An optional name for the capacity feature.

        """
        self.constant_capacity = None

        self.capacity = Feature(
            name=name, shape=("n_machines",), value_type="count"
        )
        if capacity is not None:
            if isinstance(capacity, Int):
                self.constant_capacity = int(capacity)
                storage: list[int] = []

            else:
                storage = convert_to_list(capacity, int)

            self.capacity.own_data(storage)

    def clear_capacity(self) -> None:
        """Empty the capacity list in the constraint."""
        self.capacity.own_data([])

    def set_capacity(self, machine_id: Int, capacity: Int) -> None:
        """Set the capacity for a specific machine."""
        machine = MachineID(machine_id)

        if self.constant_capacity is not None:
            raise ValueError(
                "Cannot add capacity to a batch constraint with constant capacity."
            )

        data = self.capacity.value if self.capacity.loaded else []

        extend_list(data, machine + 1, lambda: 1)
        data[machine] = int(capacity)

        self.capacity.own_data(data)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        n_machines = instance.n_machines
        cap = self.constant_capacity

        if cap is not None:
            extend_list(self.capacity.value, n_machines, lambda: cap)

        elif len(self.capacity.value) != n_machines:
            raise ValueError(
                f"Capacity list length {len(self.capacity.value)} does not match the number of machines {instance.n_machines}."
            )

    @override
    def reset(self, state: ScheduleState) -> None:
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

        self._release_times = [[MAX_TIME] for _ in range(n_machines)]
        self._free_slots = [
            [capacity] for capacity in self.capacity.value[:n_machines]
        ]

    @override
    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        self.machine_map[machine_id].discard(task_id)

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for machine in state.get_original_machines(task_id):
            self.machine_map[machine].discard(task_id)

        end_time = state.get_end_lb(task_id)
        release_times = self._release_times[machine_id]
        free_slots = self._free_slots[machine_id]

        # Same (append-only) profile as ResourceConstraint, with unit usage.
        idx = len(release_times)
        while idx > 0 and release_times[idx - 1] < end_time:
            idx -= 1

        release_times.insert(idx, end_time)
        free_slots.insert(idx, free_slots[idx - 1])

        for j in range(idx, len(free_slots)):
            free_slots[j] -= 1

        for j, slots in enumerate(free_slots):
            if slots <= 0:
                for other_task_id in self.machine_map[machine_id]:
                    state.tight_start_lb(
                        other_task_id, release_times[j], machine_id
                    )

                break

    @override
    def get_entry(self) -> str:
        if self.constant_capacity:
            return f"batch={self.constant_capacity}"

        return "batch"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "batch"

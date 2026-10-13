"""GT Tetris Backend: Giffler-Thompson conflict-set dispatching kernel."""

from heapq import heappop

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.backend import Instruction
from cpscheduler.environment.backend.tetris.tetris import TetrisBackend
from cpscheduler.environment.constants import (
    GLOBAL_MACHINE_ID,
    MAX_TIME,
    MachineID,
    TaskID,
)
from cpscheduler.environment.state import ScheduleState


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class GTTetrisBackend(TetrisBackend):
    """Giffler-Thompson dispatching kernel.

    Restricts the candidate set exposed to the policy to the conflict set of
    the critical machine, defined as the machine achieving the earliest
    possible task completion among all unlocked tasks.
    This enables generating active schedules without insertion.

    Giffler-Thompson algorithm is only correct under disjunctive resources
    (each machine processes one task at a time) and regular objective,
    this backend may not work as intended (generating active schedules) when
    used in problems with batching or shared resource.
    """

    backend = "gt_tetris"

    _cached_conflict_set: set[tuple[TaskID, MachineID]]
    _cached_critical_machine: MachineID
    _cached_generation: int

    @override
    def reset(self) -> None:
        super().reset()
        self._cached_conflict_set = set()
        self._cached_critical_machine = GLOBAL_MACHINE_ID
        self._cached_generation = -1

    def _compute_conflict_set(
        self, state: ScheduleState
    ) -> tuple[set[tuple[TaskID, MachineID]], MachineID]:
        """Return the GT conflict set as (task, machine) pairs, and the critical machine.

        Single pass over every (unlocked task, feasible machine) pair,
        caching start_lb/completion inline to avoid a second round of state
        accessor calls. Ties for critical machine are broken by iteration
        order; see the completeness caveat in the class docstring.
        """
        if state.generation == self._cached_generation:
            return self._cached_conflict_set, self._cached_critical_machine

        unlocked = state.get_unlocked_tasks()

        machine_tasks: list[list[TaskID]] = [
            [] for _ in range(state.n_machines)
        ]
        best_completion = MAX_TIME
        c_machine: MachineID = GLOBAL_MACHINE_ID

        for task_id in unlocked:
            for machine_id in state.get_machines(task_id):
                machine_tasks[machine_id].append(task_id)

                end_lb = state.get_end_lb(task_id, machine_id)

                if end_lb < best_completion:
                    best_completion = end_lb
                    c_machine = machine_id

        conflict_set: set[tuple[TaskID, MachineID]] = (
            set()
            if c_machine == GLOBAL_MACHINE_ID
            else {
                (task_id, c_machine)
                for task_id in machine_tasks[c_machine]
                if state.get_start_lb(task_id, c_machine) < best_completion
            }
        )

        self._cached_conflict_set = conflict_set
        self._cached_critical_machine = c_machine
        self._cached_generation = state.generation

        return conflict_set, c_machine

    @override
    def get_eligible_tasks(self, state: ScheduleState) -> list[TaskID]:
        conflict_set, _ = self._compute_conflict_set(state)
        return list({task_id for task_id, _ in conflict_set})

    @override
    def get_action_mask(
        self, state: ScheduleState
    ) -> dict[MachineID, set[TaskID]]:
        conflict_set, critical_machine = self._compute_conflict_set(state)

        if not conflict_set:
            return {}

        return {
            critical_machine: {
                task_id
                for task_id, machine_id in conflict_set
                if machine_id == critical_machine
            }
        }

    @override
    def dispatch_instruction(self, state: ScheduleState) -> Instruction | None:
        if not self._queue:
            return None

        _, _, instruction = heappop(self._queue)

        if not instruction.is_ready(state, self):
            raise RuntimeError(
                f"Cannot process unready instruction {instruction}."
            )

        semantic = instruction.semantic(state, self)
        if semantic and semantic["type"] == "execution":
            pair: tuple[int, int] = (semantic["task"], semantic["machine"])

            if pair not in self._compute_conflict_set(state)[0]:
                raise RuntimeError(
                    f"Cannot execute task {pair[0]} on machine {pair[1]} because "
                    "it does not belong to the current conflict set."
                )

        return instruction

    @override
    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        super().backtrack(mark, changed_tasks, state)
        self._cached_generation = state.generation

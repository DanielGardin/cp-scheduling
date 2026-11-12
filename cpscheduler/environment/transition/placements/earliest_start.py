"""Placement at the earliest start of the task, not before the clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import GLOBAL_MACHINE_ID, MAX_TIME
from cpscheduler.environment.transition.core import Placement

if TYPE_CHECKING:
    from cpscheduler.environment.constants import MachineID, TaskID, Time
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.core import CandidateRule, Clock


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class EarliestStartPlacement(Placement):
    """Start the task as early as possible, but not before the clock.

    Without a selected machine, the task goes to its candidate machine with the
    earliest start, ties broken by machine index. Without a clock, the start is
    the earliest start of the task, appending it to the partial schedule.
    """

    @override
    def place(
        self,
        state: ScheduleState,
        task_id: TaskID,
        machine_id: MachineID,
        candidates: CandidateRule,
        clock: Clock,
    ) -> tuple[MachineID, Time]:
        now = clock.time

        if machine_id != GLOBAL_MACHINE_ID:
            return machine_id, max(now, state.get_start_lb(task_id, machine_id))

        best_machine: MachineID = GLOBAL_MACHINE_ID
        best_start = MAX_TIME

        # Sorted, as the order of the feasible machines changes after
        # backtracking, and ties must not depend on the explored branches.
        for candidate in sorted(state.get_machines(task_id)):
            if not candidates.is_candidate(state, clock, task_id, candidate):
                continue

            start = max(now, state.get_start_lb(task_id, candidate))
            if start < best_start:
                best_machine = candidate
                best_start = start

        if best_machine == GLOBAL_MACHINE_ID:
            raise RuntimeError(
                f"Cannot execute task {task_id} because it is not a candidate "
                "on any machine, the candidate tasks are "
                f"{candidates.eligible_tasks(state, clock)}."
            )

        return best_machine, best_start

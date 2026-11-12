"""Candidate rule of every unlocked task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import GLOBAL_MACHINE_ID
from cpscheduler.environment.transition.core import CandidateRule

if TYPE_CHECKING:
    from cpscheduler.environment.constants import MachineID, TaskID
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.core import ActionMask, Clock


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class UnlockedCandidates(CandidateRule):
    """Every unlocked task, on every feasible machine.

    With an append placement, this is the serial schedule generation scheme
    that reaches every semi-active schedule (Kolisch, 1996).
    """

    @override
    def candidates(self, state: ScheduleState, clock: Clock) -> ActionMask:
        mask: ActionMask = {}

        for task_id in state.get_unlocked_tasks():
            for machine_id in state.get_machines(task_id):
                mask.setdefault(machine_id, set()).add(task_id)

        return mask

    @override
    def is_candidate(
        self,
        state: ScheduleState,
        clock: Clock,
        task_id: TaskID,
        machine_id: MachineID,
    ) -> bool:
        domains = state.domains

        if domains.fixed[task_id] or domains.dependencies[task_id]:
            return False

        return machine_id == GLOBAL_MACHINE_ID or domains.machines.is_feasible(
            task_id, machine_id
        )

    @override
    def eligible_tasks(
        self, state: ScheduleState, clock: Clock
    ) -> list[TaskID]:
        return state.get_unlocked_tasks()

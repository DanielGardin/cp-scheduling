"""Candidate rule of the tasks that can start at the time of the clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.transition.core import CandidateRule

if TYPE_CHECKING:
    from cpscheduler.environment.constants import MachineID, TaskID
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.core import ActionMask, Clock


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class AvailableCandidates(CandidateRule):
    """Every task that can start at the time of the clock.

    With a clock that only advances when no task is available, every task
    starts as soon as a machine can take it, which gives non-delay schedules
    (Baker, 1974).
    """

    @override
    def candidates(self, state: ScheduleState, clock: Clock) -> ActionMask:
        mask: ActionMask = {}

        for task_id in state.get_available_tasks(clock.time):
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
        return state.can_start(task_id, clock.time, machine_id)

    @override
    def eligible_tasks(
        self, state: ScheduleState, clock: Clock
    ) -> list[TaskID]:
        return state.get_available_tasks(clock.time)

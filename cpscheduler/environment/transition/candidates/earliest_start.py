"""Candidate rule of the tasks that can start the earliest."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import GLOBAL_MACHINE_ID, MAX_TIME
from cpscheduler.environment.transition.core import CandidateRule

if TYPE_CHECKING:
    from cpscheduler.environment.constants import MachineID, TaskID, Time
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.core import ActionMask, Clock

STALE_GENERATION = -1


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class EarliestStartCandidates(CandidateRule):
    """Every unlocked task that can start at the earliest start of them all.

    With an append placement, no machine is kept idle while a task could start
    on it, which gives the non-delay schedules (Baker, 1974) without a clock.
    """

    _earliest_start: Time
    _generation: int

    def __init__(self) -> None:
        self.reset()

    @override
    def reset(self) -> None:
        self._earliest_start = MAX_TIME
        self._generation = STALE_GENERATION

    def _update(self, state: ScheduleState) -> Time:
        # Every state change is propagated, which bumps the generation, so the
        # cache stays correct whatever the instructions did.
        if state.generation != self._generation:
            self._earliest_start = state.get_earliest_start_lb()
            self._generation = state.generation

        return self._earliest_start

    @override
    def candidates(self, state: ScheduleState, clock: Clock) -> ActionMask:
        earliest_start = self._update(state)
        mask: ActionMask = {}

        for task_id in state.get_unlocked_tasks():
            for machine_id in state.get_machines(task_id):
                if state.get_start_lb(task_id, machine_id) == earliest_start:
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

        if machine_id != GLOBAL_MACHINE_ID and not domains.machines.is_feasible(
            task_id, machine_id
        ):
            return False

        return state.get_start_lb(task_id, machine_id) == self._update(state)

    @override
    def eligible_tasks(
        self, state: ScheduleState, clock: Clock
    ) -> list[TaskID]:
        earliest_start = self._update(state)

        return [
            task_id
            for task_id in state.get_unlocked_tasks()
            if state.get_start_lb(task_id) == earliest_start
        ]

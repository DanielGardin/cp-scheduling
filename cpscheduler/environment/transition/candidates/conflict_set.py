"""Candidate rule of the Giffler-Thompson conflict set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy_extensions import mypyc_attr
from typing_extensions import override

from cpscheduler.environment.constants import GLOBAL_MACHINE_ID, MAX_TIME
from cpscheduler.environment.transition.core import CandidateRule

if TYPE_CHECKING:
    from cpscheduler.environment.constants import MachineID, TaskID
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.core import ActionMask, Clock

STALE_GENERATION = -1


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class ConflictSetCandidates(CandidateRule):
    """Conflict set of the critical machine (Giffler and Thompson, 1960).

    The critical machine m* is the one where an unlocked task reaches the
    earliest completion C*. The candidates are the unlocked tasks of m* that
    can start before C*. Ties for m* go to the first unlocked task, then to
    its first machine by index.

    With an append placement, this reaches exactly the active schedules when
    every machine is disjunctive and the objective is regular. On any other
    problem the rule is still well defined, but it is no longer known which
    schedules it reaches.

    The conflict set can be narrowed by `delta` (Storer et al., 1992): with
    S* the earliest start on m*, the candidates start before
    S* + delta (C* - S*), and the tasks that start at S* are always
    candidates. Every schedule reached with a `delta` is reached with a larger
    one, up to the whole conflict set at 1.
    """

    delta: float

    _conflict_set: set[TaskID]
    _critical_machine: MachineID
    _generation: int

    def __init__(self, delta: float = 1.0) -> None:
        """Initialize the rule.

        Parameters
        ----------
        delta: float, optional
            The fraction of the conflict set that is kept. Default to 1, the
            whole conflict set.

        Raises
        ------
        ValueError
            If `delta` is not between 0 and 1.

        """
        if not 0.0 <= delta <= 1.0:
            raise ValueError(
                f"Expected a delta between 0 and 1, but got {delta}."
            )

        self.delta = delta
        self.reset()

    @override
    def reset(self) -> None:
        self._conflict_set = set()
        self._critical_machine = GLOBAL_MACHINE_ID
        self._generation = STALE_GENERATION

    # Cached getter
    def _get_conflict_set(self, state: ScheduleState) -> set[TaskID]:
        if state.generation == self._generation:
            return self._conflict_set

        machine_tasks: list[list[TaskID]] = [
            [] for _ in range(state.n_machines)
        ]
        best_completion = MAX_TIME
        critical_machine: MachineID = GLOBAL_MACHINE_ID

        for task_id in state.get_unlocked_tasks():
            for machine_id in sorted(state.get_machines(task_id)):
                machine_tasks[machine_id].append(task_id)

                end_lb = state.get_end_lb(task_id, machine_id)
                if end_lb < best_completion:
                    best_completion = end_lb
                    critical_machine = machine_id

        conflict_set: set[TaskID] = set()
        if critical_machine != GLOBAL_MACHINE_ID:
            tasks = machine_tasks[critical_machine]
            delta = self.delta

            if delta == 1.0:
                conflict_set = {
                    task_id
                    for task_id in tasks
                    if state.get_start_lb(task_id, critical_machine)
                    < best_completion
                }

            else:
                starts = [
                    state.get_start_lb(task_id, critical_machine)
                    for task_id in tasks
                ]
                earliest_start = min(starts)
                threshold = float(earliest_start) + delta * float(
                    best_completion - earliest_start
                )

                conflict_set = {
                    task_id
                    for task_id, start in zip(tasks, starts, strict=True)
                    if float(start) < threshold or start == earliest_start
                }

            # The conflict set should include at least one task (the one that defines the critical machine)
            # The only exception is when the critical machine has zero-duration tasks, which are
            # all completed at their start, thus failing the threshold.
            if not conflict_set:
                conflict_set = {
                    task_id
                    for task_id in tasks
                    if state.get_remaining_time(task_id, critical_machine) == 0
                }

        self._conflict_set = conflict_set
        self._critical_machine = critical_machine
        self._generation = state.generation
        return conflict_set

    @override
    def candidates(self, state: ScheduleState, clock: Clock) -> ActionMask:
        conflict_set = self._get_conflict_set(state)

        if not conflict_set:
            return {}

        return {self._critical_machine: set(conflict_set)}

    @override
    def is_candidate(
        self,
        state: ScheduleState,
        clock: Clock,
        task_id: TaskID,
        machine_id: MachineID,
    ) -> bool:
        return task_id in self._get_conflict_set(state) and machine_id in (
            GLOBAL_MACHINE_ID,
            self._critical_machine,
        )

    @override
    def eligible_tasks(
        self, state: ScheduleState, clock: Clock
    ) -> list[TaskID]:
        return sorted(self._get_conflict_set(state))

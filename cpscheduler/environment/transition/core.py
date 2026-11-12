"""Core transition class for the MDP formulation of the scheduling problem.

A transition is composed of a clock T, a placement ψ and a candidate rule Φ.
This module defines the transition and the base class of each component, the
implementations live in `clocks`, `placements` and `candidates`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, final

from mypy_extensions import mypyc_attr
from typing_extensions import TypeVar

from cpscheduler.environment.constants import GLOBAL_MACHINE_ID, MIN_TIME
from cpscheduler.environment.mixins import EzPickle

if TYPE_CHECKING:
    from collections.abc import Callable

    from cpscheduler.environment.constants import MachineID, TaskID, Time
    from cpscheduler.environment.state import ScheduleState
    from cpscheduler.environment.transition.actions import Instruction

ActionMask = dict["MachineID", set["TaskID"]]

EventID = int


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Clock(EzPickle):
    """Clock T: the instruction queue and the reference time of placements.

    The clock decides when a queued instruction is due, and which time "now"
    is for the placement. Advancing the clock is its only effect on the state.
    """

    time: Time
    "Reference time for the placement, the earliest start of any new task."

    def reset(self) -> None:
        """Empty the queue and restart the clock."""
        self.time = MIN_TIME

    def start_episode(self, state: ScheduleState) -> None:
        """Prepare for the first decision, after the initial propagation."""

    def push(
        self,
        instruction: Instruction[Any],
        time: Time | None,
        priority: float | None,
        event_id: int,
    ) -> None:
        """Queue an instruction, `time` is interpreted by the clock."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement `push`."
        )

    def pop(
        self, state: ScheduleState, transition: Transition[Any]
    ) -> Instruction[Any] | None:
        """Return the next instruction to process, or None to end the step.

        The clock owns the queue discipline: which queued instruction is due,
        and what happens to one that is not ready, see `Instruction.blocking`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement `pop`."
        )

    def is_empty(self) -> bool:
        """Return whether no instruction is queued."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement `is_empty`."
        )

    def halt(self) -> None:
        """End the current step, the remaining instructions stay queued."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement `halt`."
        )

    def checkpoint(self, mark: int) -> None:
        """Record the clock at a backtracking checkpoint."""

    def backtrack(self, mark: int) -> None:
        """Drop every queued instruction and restore the clock at `mark`.

        Checkpoints are only taken with an empty queue, but a step that fails
        stops with its remaining instructions still queued.
        """


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class GlobalClock(Clock):
    """A clock with a single time shared by every machine.

    No task starts before its time, and instructions that need time to pass,
    such as waiting, are only defined for global clocks.
    """

    def advance_to(self, time: Time) -> None:
        """Make the untimed instructions wait until `time`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement `advance_to`."
        )


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Placement(EzPickle):
    """Placement ψ: the state change that puts a selected task in the schedule.

    `place` answers where the task goes, without changing the state, and
    `apply` commits it. The bounds of the state must reflect the placement:
    after `apply` and the propagation, the start lower bound of every other
    task is its earliest start under this placement, as the candidate rules
    and the observations read it from the state.

    Note that the placement is not a policy: it does not select a task, or
    determines the time, or machine of a task. It only computes where it would
    go if selected.

    """

    def place(
        self,
        state: ScheduleState,
        task_id: TaskID,
        machine_id: MachineID,
        candidates: CandidateRule,
        clock: Clock,
    ) -> tuple[MachineID, Time]:
        """Return the machine and start of `task_id`.

        Parameters
        ----------
        state: ScheduleState
            The current state, it must not be modified.

        task_id: TaskID
            The selected task.

        machine_id: MachineID
            The selected machine, or GLOBAL_MACHINE_ID to let the placement
            choose among the candidate machines of the task.

        candidates: CandidateRule
            The candidate rule, a chosen machine must be a candidate one.

        clock: Clock
            The clock, no task starts before its time.

        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement `place`."
        )

    def apply(
        self,
        state: ScheduleState,
        task_id: TaskID,
        machine_id: MachineID,
        start: Time,
    ) -> None:
        """Commit `task_id` on `machine_id` at `start`, changing the state.

        Only called while an instruction is processed. A placement that
        restricts where the next tasks can go tightens their bounds here.
        """
        # FUTURE: The append semantics is still forced by `MachineConstraint`.
        # Once it allows gaps, an append placement pushes here the other tasks
        # of the machine to the end of this one.
        state.assign_task(task_id, machine_id, start)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class CandidateRule(EzPickle):
    """Candidate set Φ: the task-machine pairs that the policy may select."""

    def reset(self) -> None:
        """Clear every cached result."""

    def candidates(self, state: ScheduleState, clock: Clock) -> ActionMask:
        """Return the candidate machines and their candidate tasks."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement `candidates`."
        )

    def is_candidate(
        self,
        state: ScheduleState,
        clock: Clock,
        task_id: TaskID,
        machine_id: MachineID,
    ) -> bool:
        """Return whether the task is a candidate on the machine.

        With GLOBAL_MACHINE_ID, return whether it is a candidate on any
        machine. An execution is ready exactly when its task is a candidate.

        Rules should override it when membership is cheaper than the whole
        candidate set, as it is queried on every execution.
        """
        mask = self.candidates(state, clock)

        if machine_id == GLOBAL_MACHINE_ID:
            return any(task_id in tasks for tasks in mask.values())

        tasks = mask.get(machine_id)

        return tasks is not None and task_id in tasks

    def eligible_tasks(
        self, state: ScheduleState, clock: Clock
    ) -> list[TaskID]:
        """Return the candidate tasks, the projection of the candidate set."""
        return sorted(
            {
                task_id
                for tasks in self.candidates(state, clock).values()
                for task_id in tasks
            }
        )

    def backtrack(self) -> None:
        """Invalidate the cached results after the state is restored."""
        self.reset()


C_co = TypeVar("C_co", bound=Clock, covariant=True, default=Clock)

transitions: dict[str, Callable[[], Transition[Any]]] = {}
"Transition name -> factory of the named composition."


def register_transition(
    name: str, factory: Callable[[], Transition[Any]]
) -> None:
    """Register a composition under a name accepted by the environment."""
    transitions[name] = factory


@final
@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class Transition(EzPickle, Generic[C_co]):
    """Transition class for the MDP formulation of the scheduling problem.

    A transition in a constructive scheduling problem is not uniquely defined,
    and can have different implementations, with different semantics.

    To define a valid transition, three components must be defined:
    - A clock semantics, which defines how time advances in the environment.
    - A placement operator, which determines the time and machine where a task
    is placed.
    - A candidate operator that indicates which tasks are eligible.

    The transition never changes the state, the instructions do. The built-in
    `execute` asks `place` where the task goes, which also checks that it is a
    candidate. Custom instructions may call `place` too, or change the state
    directly. In the latter case they can leave the schedule class of the
    composition: the reachable-class, candidate-membership and replay
    guarantees only hold for the built-in instructions.
    """

    name: str
    "Name of the composition, the registry name for the named ones."

    clock: C_co
    "Clock T, the instruction queue and the reference time of placements."

    placement: Placement
    "Placement ψ, the machine and start of a selected task."

    candidate: CandidateRule
    "Candidate rule Φ, the source of the action mask."

    _track_time: bool

    _next_event_id: EventID

    def __init__(
        self,
        clock: C_co,
        placement: Placement,
        candidate: CandidateRule,
        name: str = "custom",
    ) -> None:
        """Compose a transition.

        Parameters
        ----------
        clock: Clock
            The clock, which queues the instructions and defines the time.

        placement: Placement
            The placement, which reads the clock.

        candidate: CandidateRule
            The candidate rule, which reads the clock and the bounds that the
            placement leaves in the state.

        name: str, optional
            The name of the composition.

        """
        self.name = name
        self.clock = clock
        self.placement = placement
        self.candidate = candidate
        self._track_time = False

        self.reset()

    @staticmethod
    def from_register(name: str) -> Transition[Any]:
        """Return a new transition from the name of a registered composition."""
        factory = transitions.get(name)

        if factory is None:
            raise ValueError(
                f"No transition {name} registered, choose one of the "
                f"following: {', '.join(transitions)}"
            )

        return factory()

    @property
    def time(self) -> Time:
        """Time of the clock."""
        return self.clock.time

    @property
    def tracks_time(self) -> bool:
        """Whether advancing the clock notifies the time-aware components."""
        return self._track_time

    def track_time(self) -> None:
        """Make the clock notify the time-aware components when it advances.

        The notification costs a propagation per advancement, so it is off
        by default. The environment enables it when one of its constraints is
        time-aware.
        """
        self._track_time = True

    def reset(self) -> None:
        """Empty the queue, and restart the clock and the candidate rule."""
        self._next_event_id = 0

        self.clock.reset()
        self.candidate.reset()

    def start_episode(self, state: ScheduleState) -> None:
        """Prepare for the first decision, after the initial propagation."""
        self.clock.start_episode(state)

    def is_empty(self) -> bool:
        """Return whether no instruction is queued."""
        return self.clock.is_empty()

    def dispatch_instruction(
        self, state: ScheduleState
    ) -> Instruction[Any] | None:
        """Return the next instruction to process, or None to end the step."""
        return self.clock.pop(state, self)

    def get_action_mask(self, state: ScheduleState) -> ActionMask:
        """Return the candidate machines and their candidate tasks."""
        return self.candidate.candidates(state, self.clock)

    def get_eligible_tasks(self, state: ScheduleState) -> list[TaskID]:
        """Return the candidate tasks, shown as available to the observer."""
        return self.candidate.eligible_tasks(state, self.clock)

    def place(
        self,
        state: ScheduleState,
        task_id: TaskID,
        machine_id: MachineID = GLOBAL_MACHINE_ID,
    ) -> tuple[MachineID, Time]:
        """Return where the transition puts a task, without changing the state.

        Raises
        ------
        RuntimeError
            If the task is not a candidate on the placed machine.

        """
        candidate = self.candidate
        clock = self.clock

        machine_id, start = self.placement.place(
            state, task_id, machine_id, candidate, clock
        )

        if not candidate.is_candidate(state, clock, task_id, machine_id):
            raise RuntimeError(
                f"Cannot execute task {task_id} on machine {machine_id} "
                "because it does not belong to the current candidate set "
                f"{candidate.candidates(state, clock)}."
            )

        return machine_id, start

    def add_instruction(
        self,
        instruction: Instruction[Any],
        time: Time | None = None,
        priority: float | None = None,
    ) -> EventID:
        """Queue an instruction, `time` is interpreted by the clock.

        Raises
        ------
        ValueError
            If the instruction needs another clock.

        """
        clock = self.clock

        if not isinstance(clock, instruction.clock):
            raise ValueError(
                f"Instruction {instruction} needs a "
                f"{instruction.clock.__name__}, but the transition "
                f"{self.name} has a {type(clock).__name__}."
            )

        event_id = self._next_event_id
        self._next_event_id += 1

        clock.push(instruction, time, priority, event_id)

        return event_id

    def get_info(self) -> dict[str, Any]:
        """Expose the clock to the environment info."""
        return {"current_time": self.clock.time}

    def checkpoint(self, mark: int) -> None:
        """Push a backtracking checkpoint."""
        self.clock.checkpoint(mark)

    def backtrack(self, mark: int) -> None:
        """Restore the clock at `mark` and invalidate every cached result."""
        self.clock.backtrack(mark)
        self.candidate.backtrack()

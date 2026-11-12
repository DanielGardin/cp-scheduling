"""Named transitions, the compositions the environment accepts by name."""

from cpscheduler.environment.transition.candidates import (
    AvailableCandidates,
    ConflictSetCandidates,
    EarliestStartCandidates,
    UnlockedCandidates,
)
from cpscheduler.environment.transition.clocks import (
    EventClock,
    NoClock,
    UnitClock,
)
from cpscheduler.environment.transition.core import (
    Transition,
    register_transition,
)
from cpscheduler.environment.transition.placements import EarliestStartPlacement


def tetris() -> Transition[NoClock]:
    """Append every unlocked task at its earliest start, without a clock.

    Every schedule it builds is semi-active, as each task starts as early as
    the tasks placed before it allow.
    """
    return Transition(
        clock=NoClock(),
        placement=EarliestStartPlacement(),
        candidate=UnlockedCandidates(),
        name="tetris",
    )


def gt_tetris() -> Transition[NoClock]:
    """Append a task of the Giffler-Thompson conflict set, without a clock.

    It reaches exactly the active schedules when every machine is disjunctive
    and the objective is regular (Giffler and Thompson, 1960). See
    `ConflictSetCandidates`.
    """
    return Transition(
        clock=NoClock(),
        placement=EarliestStartPlacement(),
        candidate=ConflictSetCandidates(),
        name="gt_tetris",
    )


def delta_gt_tetris(delta: float) -> Transition[NoClock]:
    """Append a task of the conflict set narrowed by `delta`, without a clock.

    It interpolates between a subset of the active schedules at 0 and all of
    them at 1, where it is `gt_tetris` (Storer et al., 1992). It is not
    registered by name, as it needs its parameter.
    """
    return Transition(
        clock=NoClock(),
        placement=EarliestStartPlacement(),
        candidate=ConflictSetCandidates(delta),
        name="delta_gt_tetris",
    )


def nd_tetris() -> Transition[NoClock]:
    """Append a task that can start the earliest, without a clock.

    It reaches the non-delay schedules (Baker, 1974), the same ones as `des`
    when a single task is executed per step. See `EarliestStartCandidates`.
    """
    return Transition(
        clock=NoClock(),
        placement=EarliestStartPlacement(),
        candidate=EarliestStartCandidates(),
        name="nd_tetris",
    )


def step() -> Transition[UnitClock]:
    """Start the available tasks at the current time step of a unit clock.

    The clock advances by one after every step, so a task can be delayed by
    stepping without an action.
    """
    return Transition(
        clock=UnitClock(),
        placement=EarliestStartPlacement(),
        candidate=AvailableCandidates(),
        name="step",
    )


def des() -> Transition[EventClock]:
    """Start the available tasks at the time of a discrete event simulation.

    The clock jumps to the next time at which a task can start, so executing
    one task per step gives non-delay schedules (Baker, 1974). Waiting, timed
    and queued instructions reach the schedules beyond them, see `EventClock`.
    """
    return Transition(
        clock=EventClock(),
        placement=EarliestStartPlacement(),
        candidate=AvailableCandidates(),
        name="des",
    )


register_transition("tetris", tetris)
register_transition("gt_tetris", gt_tetris)
register_transition("nd_tetris", nd_tetris)
register_transition("step", step)
register_transition("des", des)

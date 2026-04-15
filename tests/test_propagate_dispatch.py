import pytest

from cpscheduler.environment.constants import GLOBAL_MACHINE_ID
from cpscheduler.environment.constraints.base import Constraint
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.des import Schedule
from cpscheduler.environment.env import (
    ASSIGNMENT,
    ABSENCE,
    BOUNDS_RESET,
    END_LB,
    END_UB,
    MACHINE_INFEASIBLE,
    PAUSE,
    PRESENCE,
    START_LB,
    START_UB,
    STATE_INFEASIBLE,
    SchedulingEnv,
)
from cpscheduler.environment.schedule_setup import SingleMachineSetup
from cpscheduler.environment.state.events import DomainEvent, VarFieldType


class RecordingConstraint(Constraint):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []

    def on_assignment(self, task_id: int, machine_id: int, state: ScheduleState) -> None:
        self.calls.append(("assignment", task_id, machine_id))

    def on_start_lb(self, task_id: int, machine_id: int, state: ScheduleState) -> None:
        self.calls.append(("start_lb", task_id, machine_id))

    def on_start_ub(self, task_id: int, machine_id: int, state: ScheduleState) -> None:
        self.calls.append(("start_ub", task_id, machine_id))

    def on_end_lb(self, task_id: int, machine_id: int, state: ScheduleState) -> None:
        self.calls.append(("end_lb", task_id, machine_id))

    def on_end_ub(self, task_id: int, machine_id: int, state: ScheduleState) -> None:
        self.calls.append(("end_ub", task_id, machine_id))

    def on_presence(self, task_id: int, state: ScheduleState) -> None:
        self.calls.append(("presence", task_id, None))

    def on_absence(self, task_id: int, state: ScheduleState) -> None:
        self.calls.append(("absence", task_id, None))

    def on_infeasibility(self, task_id: int, machine_id: int, state: ScheduleState) -> None:
        self.calls.append(("infeasibility", task_id, machine_id))

    def on_pause(self, task_id: int, machine_id: int, state: ScheduleState) -> None:
        self.calls.append(("pause", task_id, machine_id))

    def on_bound_reset(self, task_id: int, state: ScheduleState) -> None:
        self.calls.append(("bound_reset", task_id, None))


@pytest.mark.parametrize(
    "field, machine_id, expected",
    [
        (ASSIGNMENT, 1, ("assignment", 0, 1)),
        (START_LB, 2, ("start_lb", 0, 2)),
        (START_UB, 2, ("start_ub", 0, 2)),
        (END_LB, 2, ("end_lb", 0, 2)),
        (END_UB, 2, ("end_ub", 0, 2)),
        (PRESENCE, GLOBAL_MACHINE_ID, ("presence", 0, None)),
        (ABSENCE, GLOBAL_MACHINE_ID, ("absence", 0, None)),
        (MACHINE_INFEASIBLE, 4, ("infeasibility", 0, 4)),
        (PAUSE, 3, ("pause", 0, 3)),
        (BOUNDS_RESET, GLOBAL_MACHINE_ID, ("bound_reset", 0, None)),
    ],
)
def test_propagate_dispatches_domain_events(
    field: VarFieldType, machine_id: int, expected: tuple[str, int, int | None]
) -> None:
    recorder = RecordingConstraint()

    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=(recorder,),
        instance={"processing_time": [2]},
    )
    env.reset()

    env.state.domain_event_queue.append(DomainEvent(0, field, machine_id))

    assert env.propagate() is True
    assert recorder.calls == [expected]
    assert env.event_count == 1
    assert env.state.domain_event_queue == []


def test_propagate_returns_false_for_state_infeasible_event() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [2]},
    )
    env.reset()
    env.state.infeasible = True

    env.state.domain_event_queue.append(DomainEvent(0, STATE_INFEASIBLE))

    assert env.propagate() is False
    assert env.event_count == 0
    assert len(env.state.domain_event_queue) == 1
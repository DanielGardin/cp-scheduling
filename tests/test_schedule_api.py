import pytest

from cpscheduler.environment.des import Schedule
from cpscheduler.environment.des.events import (
    AdvanceTimeEvent,
    CheckpointEvent,
    ExecuteEvent,
    SubmitEvent,
)
from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.schedule_setup import (
    IdenticalParallelMachineSetup,
    SingleMachineSetup,
)


def _single_env() -> SchedulingEnv:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [2, 3]},
    )
    env.reset()
    return env


def _parallel_env() -> SchedulingEnv:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(n_machines=2, disjunctive=False),
        instance={"processing_time": [2, 3]},
    )
    env.reset()
    return env


def test_schedule_remove_reschedule_and_clear() -> None:
    env = _single_env()
    schedule = Schedule()

    timed = CheckpointEvent()
    untimed = SubmitEvent(0)
    schedule.add_event(timed, env.state, time=5)
    schedule.add_event(untimed, env.state)

    schedule.remove_event(timed)
    assert list(schedule.peek_events_at_time(5)) == []

    schedule.reschedule_event(untimed, 4)
    assert list(schedule.peek_events_at_time(0)) == []
    assert [type(event) for event in schedule.peek_events_at_time(4)] == [SubmitEvent]

    schedule.clear_schedule()
    assert schedule.is_empty()


def test_schedule_change_priority_reorders_non_timed_events() -> None:
    env = _parallel_env()
    schedule = Schedule()

    low = SubmitEvent(0)
    high = SubmitEvent(1)
    schedule.add_event(low, env.state)
    schedule.add_event(high, env.state)

    schedule.change_event_priority(high, 10)

    ordered = list(schedule.peek_events_at_time(env.state.time))

    assert ordered[0] is high
    assert ordered[1] is low


def test_schedule_change_priority_rejects_timed_events() -> None:
    env = _single_env()
    schedule = Schedule()

    event = CheckpointEvent()
    schedule.add_event(event, env.state, time=3)

    with pytest.raises(ValueError, match="is not scheduled"):
        schedule.change_event_priority(event, 1)


def test_schedule_remove_event_rejects_missing_event() -> None:
    with pytest.raises(KeyError):
        Schedule().remove_event(AdvanceTimeEvent(1))


def test_schedule_add_event_resolves_machine_and_peeks() -> None:
    env = _single_env()
    schedule = Schedule()

    event = ExecuteEvent(0)
    schedule.add_event(event, env.state)

    assert event.machine_id == 0
    assert [type(item) for item in schedule.peek_events()] == [ExecuteEvent]

import pytest

from cpscheduler.environment.constants import GLOBAL_MACHINE_ID
from cpscheduler.environment.des import SimulationEvent, parse_instruction, SingleAction
from cpscheduler.environment.des.base import Schedule, instructions
from cpscheduler.environment.des.events import (
    AdvanceTimeEvent,
    CheckpointEvent,
    CompleteEvent,
    ExecuteEvent,
    PauseEvent,
    ResumeEvent,
    SubmitEvent,
)
from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.constraints import ReleaseDateConstraint
from cpscheduler.environment.schedule_setup import (
    IdenticalParallelMachineSetup,
    SingleMachineSetup,
)


def _single_task_env_single_machine(processing_time: int = 5) -> SchedulingEnv:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        instance={"processing_time": [processing_time]},
    )
    env.reset()
    return env


def _single_task_env_two_machines(processing_time: int = 5) -> SchedulingEnv:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(n_machines=2, disjunctive=False),
        instance={"processing_time": [processing_time]},
    )
    env.reset()
    return env


def test_instruction_registry_contains_all_des_instructions() -> None:
    expected = {
        "execute": ExecuteEvent,
        "submit": SubmitEvent,
        "pause": PauseEvent,
        "resume": ResumeEvent,
        "complete": CompleteEvent,
        "advance": AdvanceTimeEvent,
    }

    assert all(name in instructions for name in expected)
    assert {name: instructions[name] for name in expected} == expected


def test_parse_instruction_builds_every_instruction_event() -> None:
    cases: list[tuple[SingleAction, type[SimulationEvent], tuple[int, ...], int| None]]  = [
        (("execute", 0), ExecuteEvent, (0, GLOBAL_MACHINE_ID), None),
        (("submit", 0, 1), SubmitEvent, (0, 1), None),
        (("pause", 0), PauseEvent, (0,), None),
        (("resume", 0), ResumeEvent, (0,), None),
        (("complete", 0), CompleteEvent, (0,), None),
        (("advance", 3), AdvanceTimeEvent, (3,), None),
        ((12, "execute", 0, 0), ExecuteEvent, (0, 0), 12),
    ]

    for raw_action, expected_cls, expected_args, expected_time in cases:
        event, time, priority = parse_instruction(raw_action)
        assert isinstance(event, expected_cls)
        assert event.args == expected_args
        assert time == expected_time
        assert priority is None  # default priority (None -> Schedule converts to 0)


def test_parse_instruction_accepts_event_instance_directly() -> None:
    event = PauseEvent(0)

    parsed, time, priority = parse_instruction(event)

    assert parsed is event
    assert time is None
    assert priority is None


def test_execute_resolve_sets_single_machine_when_global_machine() -> None:
    env = _single_task_env_single_machine()
    state = env.state

    event = ExecuteEvent(task_id=0)
    event.resolve(state)

    assert event.machine_id == 0


def test_execute_resolve_rejects_invalid_machine() -> None:
    env = _single_task_env_two_machines()

    event = ExecuteEvent(task_id=0, machine_id=99)

    with pytest.raises(ValueError, match="is not available for task"):
        event.resolve(env.state)


def test_execute_resolve_keeps_explicit_valid_machine() -> None:
    env = _single_task_env_two_machines()
    event = ExecuteEvent(task_id=0, machine_id=1)

    event.resolve(env.state)

    assert event.machine_id == 1


def test_execute_earliest_ready_and_process() -> None:
    env = _single_task_env_single_machine()
    state = env.state

    event = ExecuteEvent(task_id=0)
    event.resolve(state)

    assert event.earliest_time(state) == 0
    assert event.is_ready(state)

    event.process(state, env.schedule)

    assert state.is_executing(0)
    assert state.runtime.get_assignment(0) == 0


def test_execute_not_ready_before_start_lb() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [5], "release_time": [4]},
    )
    env.reset()
    state = env.state

    event = ExecuteEvent(task_id=0)
    event.resolve(state)

    assert event.earliest_time(state) == 4
    assert not event.is_ready(state)


def test_submit_behaves_like_non_blocking_execute() -> None:
    env = _single_task_env_single_machine()
    state = env.state

    event = SubmitEvent(task_id=0)
    event.resolve(state)

    assert event.blocking is False
    assert event.is_ready(state)

    event.process(state, env.schedule)

    assert state.is_executing(0)


def test_submit_not_ready_before_release_lb() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [5], "release_time": [4]},
    )
    env.reset()

    event = SubmitEvent(task_id=0)
    event.resolve(env.state)

    assert event.earliest_time(env.state) == 4
    assert not event.is_ready(env.state)


def test_pause_readiness_and_process() -> None:
    env = _single_task_env_single_machine(processing_time=10)
    state = env.state

    execute = ExecuteEvent(task_id=0)
    execute.resolve(state)
    execute.process(state, env.schedule)

    pause = PauseEvent(task_id=0)

    assert pause.is_ready(state)

    state.advance_time_(3)
    pause.process(state, env.schedule)

    assert state.is_paused(0)
    assert not state.is_executing(0)
    assert state.runtime.get_end(0) == 3


def test_pause_not_ready_when_task_not_executing() -> None:
    env = _single_task_env_single_machine()

    event = PauseEvent(task_id=0)

    assert not event.is_ready(env.state)


def test_resume_readiness_and_process() -> None:
    env = _single_task_env_single_machine(processing_time=10)
    state = env.state

    execute = ExecuteEvent(task_id=0)
    execute.resolve(state)
    execute.process(state, env.schedule)

    state.advance_time_(4)
    PauseEvent(task_id=0).process(state, env.schedule)

    event = ResumeEvent(task_id=0)

    assert event.is_ready(state)

    last_machine_before_resume = state.runtime.get_assignment(0)
    history_len_before = len(state.runtime.history[0])

    event.process(state, env.schedule)

    assert state.is_executing(0)
    assert state.runtime.get_assignment(0) == last_machine_before_resume
    assert len(state.runtime.history[0]) == history_len_before + 1


def test_resume_not_ready_when_task_not_paused() -> None:
    env = _single_task_env_single_machine()

    event = ResumeEvent(task_id=0)

    assert not event.is_ready(env.state)


def test_complete_readiness_and_process_schedules_checkpoint_at_end() -> None:
    env = _single_task_env_single_machine(processing_time=6)
    state = env.state
    schedule = Schedule()

    execute = ExecuteEvent(task_id=0)
    execute.resolve(state)
    execute.process(state, schedule)

    event = CompleteEvent(task_id=0)

    assert event.is_ready(state)

    expected_end = state.get_end_lb(0)
    event.process(state, schedule)

    assert schedule.next_time() == expected_end

    timed_events = schedule._timed_events[expected_end]
    assert len(timed_events) == 1
    assert isinstance(timed_events[0], CheckpointEvent)


def test_complete_not_ready_when_task_not_executing() -> None:
    env = _single_task_env_single_machine()

    event = CompleteEvent(task_id=0)

    assert not event.is_ready(env.state)


def test_advance_process_schedules_checkpoint_after_time_delta() -> None:
    env = _single_task_env_single_machine()
    state = env.state
    schedule = Schedule()

    state.advance_time_(2)

    event = AdvanceTimeEvent(5)
    event.process(state, schedule)

    target_time = 7

    assert schedule.next_time() == target_time

    timed_events = schedule._timed_events[target_time]
    assert len(timed_events) == 1
    assert isinstance(timed_events[0], CheckpointEvent)


def test_schedule_add_event_resolves_execute_machine() -> None:
    env = _single_task_env_single_machine()
    schedule = Schedule()
    event = ExecuteEvent(0)

    schedule.add_event(event, env.state)

    assert event.machine_id == 0


def test_schedule_add_event_rejects_past_time() -> None:
    env = _single_task_env_single_machine()
    schedule = Schedule()

    env.state.advance_time_(3)

    with pytest.raises(ValueError, match="Cannot schedule event in the past"):
        schedule.add_event(AdvanceTimeEvent(1), env.state, time=2)


def test_advance_clock_prefers_schedule_time_over_unlocked_bounds() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [2], "release_time": [10]},
    )
    env.reset()

    env.schedule.add_event(CheckpointEvent(), env.state, time=3)

    advanced_with_events = env.advance_clock()

    assert advanced_with_events is True
    assert env.state.time == 3


def test_advance_clock_uses_next_start_lb_when_schedule_empty() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [2], "release_time": [4]},
    )
    env.reset()

    advanced_with_events = env.advance_clock()

    assert advanced_with_events is False
    assert env.state.time == 4


def test_non_blocking_not_ready_event_is_deferred() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [2, 2], "release_time": [3, 0]},
    )
    env.reset()

    schedule = Schedule()
    schedule.add_event(SubmitEvent(0), env.state)
    schedule.add_event(ExecuteEvent(1), env.state)

    queued = list(schedule.instruction_queue(env.state))

    assert len(queued) == 1
    assert isinstance(queued[0], ExecuteEvent)

    assert schedule.next_time() == 3
    deferred = schedule._non_timed_events[3]
    assert len(deferred) == 1
    assert isinstance(deferred[0][3], SubmitEvent)

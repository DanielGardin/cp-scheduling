import pytest

from cpscheduler.environment.constraints import ReleaseDateConstraint
from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.setups import (
    IdenticalParallelMachineSetup,
    SingleMachineSetup,
)
from cpscheduler.environment.transition import (
    EventClock,
    ExecuteInstruction,
    SubmitInstruction,
)
from cpscheduler.environment.transition.actions import instructions
from tests.conftest import env_setup


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
        "execute": ExecuteInstruction,
        "submit": SubmitInstruction,
    }

    assert all(name in instructions for name in expected)
    assert {name: instructions[name] for name in expected} == expected


def test_execute_resolve_sets_single_machine_when_global_machine() -> None:
    env = _single_task_env_single_machine()
    state = env.state

    event = ExecuteInstruction(task_id=0)
    resolved_event = event.resolve(state)

    assert resolved_event.machine_id == 0


def test_execute_resolve_rejects_invalid_machine() -> None:
    env = _single_task_env_two_machines()

    event = ExecuteInstruction(task_id=0, machine_id=99)

    with pytest.raises(ValueError, match="is not available for task"):
        event.resolve(env.state)


def test_execute_resolve_keeps_explicit_valid_machine() -> None:
    env = _single_task_env_two_machines()
    event = ExecuteInstruction(task_id=0, machine_id=1)

    event.resolve(env.state)

    assert event.machine_id == 1


def test_execute_earliest_ready_and_process() -> None:
    env = _single_task_env_single_machine()
    state = env.state
    transition = env.transition

    assert isinstance(transition.clock, EventClock)

    event = ExecuteInstruction(task_id=0)
    event.resolve(state)

    assert event.earliest_time(state) == 0
    assert event.is_ready(state, transition)

    event.process(state, transition)

    assert state.get_start(0) == 0
    assert state.get_assignment(0) == 0


def test_execute_not_ready_before_start_lb() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        # The second task keeps the first decision point at time 0.
        instance={"processing_time": [5, 1], "release_time": [4, 0]},
    )
    env.reset()
    state = env.state
    transition = env.transition

    assert isinstance(transition.clock, EventClock)

    event = ExecuteInstruction(task_id=0)
    event.resolve(state)

    assert event.earliest_time(state) == 4
    assert not event.is_ready(state, transition)


def test_submit_behaves_like_non_blocking_execute() -> None:
    env = _single_task_env_single_machine()
    state = env.state
    transition = env.transition

    assert isinstance(transition.clock, EventClock)

    event = SubmitInstruction(task_id=0)
    event.resolve(state)

    assert event.blocking is False
    assert event.is_ready(state, transition)

    event.process(state, transition)

    assert state.get_start(0) == transition.time


def test_submit_not_ready_before_release_lb() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        # The second task keeps the first decision point at time 0.
        instance={"processing_time": [5, 1], "release_time": [4, 0]},
    )
    state = env.state
    transition = env.transition

    assert isinstance(transition.clock, EventClock)

    env.reset()

    event = SubmitInstruction(task_id=0)
    event.resolve(env.state)

    assert event.earliest_time(state) == 4
    assert not event.is_ready(state, transition)


def test_non_blocking_not_ready_event_is_deferred() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [2, 2], "release_time": [3, 0]},
    )
    env.reset()

    transition = env.transition
    assert isinstance(transition.clock, EventClock)

    transition.add_instruction(SubmitInstruction(0))
    transition.add_instruction(ExecuteInstruction(1))

    event = transition.dispatch_instruction(env.state)

    assert isinstance(event, ExecuteInstruction)


def test_clean_cache_after_run() -> None:
    env = env_setup("ta01")

    env.reset()
    env.step([("submit", i) for i in range(env.state.n_tasks)])

    assert isinstance(env.transition.clock, EventClock)
    assert env.transition.is_empty()

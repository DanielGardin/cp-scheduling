import pytest

from cpscheduler.environment.backend.actions import instructions
from cpscheduler.environment.backend.des import (
    DESBackend,
    ExecuteEvent,
    SubmitEvent,
)
from cpscheduler.environment.constraints import ReleaseDateConstraint
from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.setups import (
    IdenticalParallelMachineSetup,
    SingleMachineSetup,
)
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
        "execute": ExecuteEvent,
        "submit": SubmitEvent,
    }

    assert all(name in instructions["des"] for name in expected)
    assert {name: instructions["des"][name] for name in expected} == expected


def test_execute_resolve_sets_single_machine_when_global_machine() -> None:
    env = _single_task_env_single_machine()
    state = env.state

    event = ExecuteEvent(task_id=0)
    resolved_event = event.resolve(state)

    assert resolved_event.machine_id == 0


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
    backend = env.backend

    assert isinstance(backend, DESBackend)

    event = ExecuteEvent(task_id=0)
    event.resolve(state)

    assert event.earliest_time(state) == 0
    assert event.is_ready(state, backend)

    event.process(state, backend)

    assert state.get_start(0) == 0
    assert state.get_assignment(0) == 0


def test_execute_not_ready_before_start_lb() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [5], "release_time": [4]},
    )
    env.reset()
    state = env.state
    backend = env.backend

    assert isinstance(backend, DESBackend)

    event = ExecuteEvent(task_id=0)
    event.resolve(state)

    assert event.earliest_time(state) == 4
    assert not event.is_ready(state, backend)


def test_submit_behaves_like_non_blocking_execute() -> None:
    env = _single_task_env_single_machine()
    state = env.state
    backend = env.backend

    assert isinstance(backend, DESBackend)

    event = SubmitEvent(task_id=0)
    event.resolve(state)

    assert event.blocking is False
    assert event.is_ready(state, backend)

    event.process(state, backend)

    assert state.get_start(0) == backend.time


def test_submit_not_ready_before_release_lb() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [5], "release_time": [4]},
    )
    state = env.state
    backend = env.backend

    assert isinstance(backend, DESBackend)

    env.reset()

    event = SubmitEvent(task_id=0)
    event.resolve(env.state)

    assert event.earliest_time(state) == 4
    assert not event.is_ready(state, backend)


def test_non_blocking_not_ready_event_is_deferred() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ReleaseDateConstraint("release_time")],
        instance={"processing_time": [2, 2], "release_time": [3, 0]},
    )
    env.reset()

    backend = env.backend
    assert isinstance(backend, DESBackend)

    backend.add_instruction(SubmitEvent(0))
    backend.add_instruction(ExecuteEvent(1))

    event = backend.dispatch_instruction(env.state)

    assert isinstance(event, ExecuteEvent)


def test_clean_cache_after_run() -> None:
    env = env_setup("ta01")

    env.reset()
    env.step([("submit", i) for i in range(env.state.n_tasks)])

    assert isinstance(env.backend, DESBackend)
    assert env.backend.is_empty()

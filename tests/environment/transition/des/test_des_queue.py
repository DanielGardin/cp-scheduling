import pytest

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    JobShopSetup,
    Makespan,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
)
from cpscheduler.environment.transition import (
    EventClock,
    ExecuteInstruction,
    HaltInstruction,
    Transition,
)
from cpscheduler.environment.transition.presets import des


def _released(
    release_times: list[int],
) -> tuple[SchedulingEnv, Transition[EventClock]]:
    transition = des()
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(len(release_times), disjunctive=False),
        [ReleaseDateConstraint()],
        Makespan(),
        transition=transition,
        instance={
            "processing_time": [1] * len(release_times),
            "release_time": release_times,
        },
    )
    env.reset()

    return env, transition


def _single_machine(
    processing_time: list[int],
) -> tuple[SchedulingEnv, Transition[EventClock]]:
    transition = des()
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        transition=transition,
        instance={"processing_time": processing_time},
    )
    env.reset()

    return env, transition


def _chain() -> tuple[SchedulingEnv, Transition[EventClock]]:
    transition = des()
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=transition,
        instance={
            "job": [0, 0],
            "operation": [0, 1],
            "machine": [0, 1],
            "processing_time": [2, 3],
        },
    )
    env.reset()

    return env, transition


@pytest.mark.parametrize("instruction", ["execute", "submit"])
def test_locked_instruction_waits_for_the_event_that_unlocks_it(
    instruction: str,
) -> None:
    # Task 1 stays locked past its start bound, until task 0 runs at 5
    env, transition = _chain()

    env.step([(instruction, 1), (5, "execute", 0)])

    assert env.state.get_start(0) == 5
    assert env.state.get_start(1) == 7
    assert transition.is_empty()


@pytest.mark.parametrize(
    "action",
    [
        [("execute", 1), ("execute", 0)],
        [("submit", 1)],
    ],
)
def test_locked_instruction_without_an_unlocking_event_is_a_deadlock(
    action: list[tuple[str, int]],
) -> None:
    env, _ = _chain()

    with pytest.raises(RuntimeError, match="Deadlock at time"):
        env.step(action)


def test_submit_unlocked_later_in_the_same_time_slot_runs_at_that_time() -> (
    None
):
    # Task 0 has zero duration, so task 1 can start at 0 once it is placed
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        transition=des(),
        instance={
            "job": [0, 0],
            "operation": [0, 1],
            "machine": [0, 1],
            "processing_time": [0, 3],
        },
    )
    env.reset()

    env.step([("submit", 1), ("submit", 0)])

    assert env.state.get_start(0) == 0
    assert env.state.get_start(1) == 0
    assert env.transition.is_empty()


def test_deferred_submit_is_not_duplicated_by_a_blocking_execute() -> None:
    # Task 1 is deferred by itself, and again by the blocked execute behind it
    env, transition = _released([0, 5, 3])

    env.step([("submit", 1), ("execute", 2), ("execute", 0)])

    assert env.state.get_start(1) == 5
    assert env.state.get_start(2) == 3
    assert env.state.get_start(0) == 3
    assert transition.is_empty()


def test_deferred_events_run_once() -> None:
    env, transition = _released([0, 5, 7, 3])

    env.step([("submit", 1), ("submit", 2), ("execute", 3), ("execute", 0)])

    assert [env.state.get_start(t) for t in range(4)] == [3, 5, 7, 3]
    assert transition.is_empty()


def test_first_decision_point_has_an_available_task() -> None:
    # Otherwise masks are all zeros and rules fail on the very first decision
    env, transition = _released([5, 8])

    assert transition.time == 5
    assert env.get_action_support() == [0]


def test_removed_event_at_the_current_time_is_not_dispatched() -> None:
    env, transition = _single_machine([1, 1])

    transition.clock.remove_event(
        transition.add_instruction(ExecuteInstruction(0))
    )
    env.step(("execute", 1))

    assert env.state.get_assigned_tasks() == [1]


def test_removed_future_event_does_not_move_the_clock() -> None:
    env, transition = _single_machine([1, 1])

    transition.clock.remove_event(
        transition.add_instruction(HaltInstruction(), time=1000)
    )

    assert transition.is_empty()

    env.step()

    assert transition.time == 0


def test_rescheduled_event_does_not_stop_at_its_old_time() -> None:
    env, transition = _single_machine([10, 1])

    transition.clock.reschedule_event(
        transition.add_instruction(HaltInstruction(), time=5), 7
    )
    env.step(("submit", 0))

    assert transition.time == 7


def test_clock_does_not_jump_to_infinity_when_nothing_can_start() -> None:
    env, transition = _single_machine([1])

    env.state.add_dependency(0, "never_resolved")
    env.propagate()
    env.step()

    assert transition.time == 0

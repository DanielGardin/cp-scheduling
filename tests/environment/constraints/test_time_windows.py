import pytest

from cpscheduler.environment import (
    DeadlineConstraint,
    IdenticalParallelMachineSetup,
    Makespan,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
)
from tests.schedule_checks import TRANSITIONS


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_task_can_end_exactly_at_its_deadline(transition: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [DeadlineConstraint(due_dates=[3])],
        Makespan(),
        transition=transition,
        instance={"processing_time": [3]},
    )
    env.reset()

    assert env.get_action_support() == [0]

    *_, terminated, truncated, _ = env.step(("execute", 0))

    assert (terminated, truncated) == (True, False)
    assert env.state.get_end(0) == 3


@pytest.mark.parametrize("transition", ["des", "step"])
def test_missing_a_start_window_ends_the_episode(transition: str) -> None:
    # A task nobody can start anymore must not keep the episode open forever
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2, disjunctive=False),
        [DeadlineConstraint(due_dates=[1, 50]), ReleaseDateConstraint()],
        Makespan(),
        transition=transition,
        instance={"processing_time": [1, 1], "release_time": [0, 5]},
    )
    env.reset()

    truncated = False
    for _ in range(10):
        *_, truncated, _ = env.step(None if transition == "step" else ("noop",))

        if truncated:
            break

    assert truncated
    assert env.state.infeasible

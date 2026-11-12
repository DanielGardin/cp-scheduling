from collections.abc import Sequence
from random import Random

import pytest

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    Makespan,
    NoWaitConstraint,
    ORPrecedenceConstraint,
    PrecedenceConstraint,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
)
from cpscheduler.environment.constants import Int
from tests.schedule_checks import (
    TRANSITIONS,
    optional_parallel_env,
    run_random_episode,
)


def _or_env(
    transition: str = "des", processing: Sequence[int] = (5, 1, 1)
) -> SchedulingEnv:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(3, disjunctive=False),
        [ORPrecedenceConstraint({2: [0, 1]})],
        Makespan(),
        transition=transition,
        instance={"processing_time": list(processing)},
    )
    env.reset()

    return env


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_no_wait_child_starts_when_the_parent_ends(transition: str) -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2, disjunctive=False),
        [NoWaitConstraint({1: [0]})],
        Makespan(),
        transition=transition,
        instance={"processing_time": [3, 2]},
    )
    run_random_episode(env, Random(0))

    assert env.state.remaining_tasks == 0
    assert env.state.get_start(1) == env.state.get_end(0)


def test_no_wait_delays_the_parent_of_a_late_child() -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2, disjunctive=False),
        [NoWaitConstraint({1: [0]}), ReleaseDateConstraint()],
        Makespan(),
        instance={"processing_time": [3, 2], "release_time": [0, 10]},
    )
    env.reset()

    env.step([("execute", 0), ("execute", 1)])

    assert (env.state.get_start(0), env.state.get_start(1)) == (7, 10)


def test_no_wait_child_without_a_start_time_is_infeasible() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [NoWaitConstraint({1: [0]})],
        Makespan(),
        instance={"processing_time": [3, 2, 5]},
    )
    env.reset()

    env.step(("execute", 0))

    assert not env.state.infeasible

    # Task 2 takes the machine at 3, when the child must start
    *_, truncated, _ = env.step(("execute", 2))

    assert truncated
    assert env.state.infeasible


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_or_child_is_locked_until_a_parent_is_scheduled(
    transition: str,
) -> None:
    env = _or_env(transition)

    assert env.state.is_locked(2)
    assert 2 not in env.get_action_support()


@pytest.mark.env
@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("transition", TRANSITIONS)
def test_or_child_starts_after_the_first_parent_ends(
    transition: str, seed: int
) -> None:
    rng = Random(seed)
    env = _or_env(transition, [rng.randint(1, 9) for _ in range(3)])

    run_random_episode(env, rng)

    assert env.state.remaining_tasks == 0
    assert env.state.get_start(2) >= min(
        env.state.get_end(0), env.state.get_end(1)
    )


def test_or_child_does_not_wait_for_the_other_parent() -> None:
    env = _or_env()

    env.step(("execute", 1))

    assert env.transition.get_info()["current_time"] == 0
    assert 2 not in env.get_action_support()

    env.step(("noop",))

    assert env.transition.get_info()["current_time"] == 1
    assert 2 in env.get_action_support()

    env.step(("execute", 2))

    assert env.state.get_start(2) == 1
    assert not env.state.is_fixed(0)


def test_rejecting_an_or_parent_forces_nothing_while_alternatives_remain() -> (
    None
):
    env = optional_parallel_env(ORPrecedenceConstraint({2: [0, 1]}), 3)

    env.step(("reject", 0))

    assert not env.state.is_absent(2)
    assert not env.state.is_present(1)

    env.state.require_task(2)
    env.propagate()

    assert env.state.is_present(1)


def test_required_or_child_requires_no_parent_with_alternatives() -> None:
    env = optional_parallel_env(ORPrecedenceConstraint({2: [0, 1]}), 3)

    env.state.require_task(2)
    env.propagate()

    assert not env.state.is_present(0)
    assert not env.state.is_present(1)


def test_or_child_without_parents_is_rejected() -> None:
    env = optional_parallel_env(ORPrecedenceConstraint({2: [0, 1]}), 3)

    env.step([("reject", 0), ("reject", 1)])

    assert env.state.is_absent(2)
    assert env.state.is_terminal()
    assert not env.state.infeasible


@pytest.mark.parametrize(
    ("precedence", "intree", "outtree", "entry"),
    [
        ({1: [0], 2: [0]}, False, True, "outtree"),
        ({2: [0, 1]}, True, False, "intree"),
        ({1: [0], 2: [1]}, True, True, "chains"),
    ],
)
def test_tree_classification(
    precedence: dict[Int, list[Int]], intree: bool, outtree: bool, entry: str
) -> None:
    constraint = PrecedenceConstraint(precedence)

    assert constraint.is_intree() == intree
    assert constraint.is_outtree() == outtree
    assert constraint.get_entry() == entry


def test_precedence_with_an_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="refers to task 5"):
        SchedulingEnv(
            SingleMachineSetup(),
            [PrecedenceConstraint({5: [0]})],
            Makespan(),
            instance={"processing_time": [1, 1]},
        )


def _two_episodes(constraint: PrecedenceConstraint) -> list[int]:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        [constraint],
        Makespan(),
        instance={"processing_time": [2, 3, 1]},
    )

    starts = []
    for _ in range(2):
        env.reset()
        env.step([("execute", 0), ("execute", 1), ("execute", 2)])
        starts.append(env.state.get_start(2))

    return starts


def test_added_precedences_hold_in_every_episode() -> None:
    constraint = PrecedenceConstraint()
    constraint.add_precedence(1, 2)

    assert _two_episodes(constraint) == [3, 3]


def test_added_chains_hold_in_every_episode() -> None:
    constraint = PrecedenceConstraint()
    constraint.add_chain([0, 1, 2])

    assert _two_episodes(constraint) == [5, 5]


def test_removed_precedences_stay_removed() -> None:
    constraint = PrecedenceConstraint({2: [1]})
    constraint.remove_precedence(1, 2)

    assert _two_episodes(constraint) == [0, 0]

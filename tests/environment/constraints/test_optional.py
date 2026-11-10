import pytest

from cpscheduler.environment import (
    AllOrNothingConstraint,
    AtMostOneConstraint,
    ExactlyOneConstraint,
    IdenticalParallelMachineSetup,
    Makespan,
    OptionalityConstraint,
    RejectableConstraint,
    SchedulingEnv,
    SingleMachineSetup,
)
from cpscheduler.environment.constraints import Constraint
from cpscheduler.environment.instance import ProblemInstance
from tests.schedule_checks import BACKENDS, optional_parallel_env


class _OptionalTasks(Constraint):
    def __init__(self, tasks: list[int]) -> None:
        self.tasks = tasks

    def initialize(self, instance: ProblemInstance) -> None:
        for task_id in self.tasks:
            instance.set_optionality(task_id)


def _group_env(group: Constraint, optional: list[int]) -> SchedulingEnv:
    return SchedulingEnv(
        IdenticalParallelMachineSetup(3, disjunctive=False),
        [group, _OptionalTasks(optional)],
        Makespan(),
        instance={"processing_time": [1, 1, 1]},
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_optional_task_can_be_rejected(backend: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [OptionalityConstraint()],
        Makespan(),
        backend=backend,
        instance={"processing_time": [5, 100]},
    )
    env.reset()

    *_, terminated, truncated, _ = env.step([("reject", 1), ("execute", 0)])

    assert (terminated, truncated) == (True, False)
    assert env.objective.value == 5


@pytest.mark.parametrize("backend", ["des", "tetris"])
def test_rejecting_a_required_task_is_infeasible(backend: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        backend=backend,
        instance={"processing_time": [5]},
    )
    env.reset()

    *_, truncated, _ = env.step(("reject", 0))

    assert truncated
    assert env.state.infeasible


def test_exactly_one_rejects_the_alternatives_of_the_chosen_task() -> None:
    env = optional_parallel_env(ExactlyOneConstraint([[0, 1, 2]]), 3)

    env.step(("execute", 0))

    assert env.state.is_absent(1)
    assert env.state.is_absent(2)
    assert env.state.is_terminal()
    assert not env.state.infeasible


def test_exactly_one_groups_are_independent() -> None:
    env = optional_parallel_env(ExactlyOneConstraint([[0, 1], [2, 3]]), 4)

    env.step(("execute", 0))

    assert env.state.is_absent(1)
    assert not env.state.is_fixed(2)
    assert not env.state.is_fixed(3)


def test_exactly_one_requires_the_last_alternative() -> None:
    env = optional_parallel_env(ExactlyOneConstraint([[0, 1, 2]]), 3)

    env.step([("reject", 0), ("reject", 1)])

    assert env.state.is_present(2)
    assert not env.state.infeasible


def test_exactly_one_without_alternatives_is_infeasible() -> None:
    env = optional_parallel_env(ExactlyOneConstraint([[0, 1]]), 2)

    env.state.forbid_task(0)
    env.state.forbid_task(1)
    env.propagate()

    assert env.state.infeasible


def test_at_most_one_allows_rejecting_every_task() -> None:
    env = optional_parallel_env(AtMostOneConstraint([[0, 1, 2]]), 3)

    env.step([("reject", 0), ("reject", 1), ("reject", 2)])

    assert env.state.is_terminal()
    assert not env.state.infeasible


@pytest.mark.parametrize("group", [AtMostOneConstraint, ExactlyOneConstraint])
def test_a_required_task_rejects_its_alternatives_on_reset(
    group: type[AtMostOneConstraint],
) -> None:
    env = _group_env(group([[0, 1, 2]]), optional=[1, 2])

    env.reset()

    assert env.state.is_absent(1)
    assert env.state.is_absent(2)
    assert not env.state.infeasible


@pytest.mark.parametrize("group", [AtMostOneConstraint, ExactlyOneConstraint])
def test_two_required_alternatives_are_infeasible(
    group: type[AtMostOneConstraint],
) -> None:
    env = _group_env(group([[0, 1]]), optional=[2])

    with pytest.raises(RuntimeError, match="stale state"):
        env.reset()


def test_at_most_one_of_nothing_holds() -> None:
    env = _group_env(AtMostOneConstraint([[]]), optional=[0, 1, 2])

    env.reset()

    assert not env.state.infeasible


def test_exactly_one_of_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty set of tasks"):
        _group_env(ExactlyOneConstraint([[]]), optional=[0, 1, 2])


def test_all_or_nothing_requires_the_group_of_a_scheduled_task() -> None:
    env = optional_parallel_env(AllOrNothingConstraint([[0, 1], [2, 3]]), 4)

    env.step(("execute", 0))

    assert env.state.is_present(1)
    assert not env.state.is_fixed(2)
    assert not env.state.is_fixed(3)


def test_all_or_nothing_rejects_the_group_of_a_rejected_task() -> None:
    env = optional_parallel_env(AllOrNothingConstraint([[0, 1], [2, 3]]), 4)

    env.step(("reject", 3))

    assert env.state.is_absent(2)
    assert env.state.is_fixed(2)
    assert not env.state.is_fixed(0)
    assert not env.state.infeasible


def test_all_or_nothing_requires_the_group_of_a_required_task_on_reset() -> (
    None
):
    env = _group_env(AllOrNothingConstraint([[0, 1]]), optional=[1, 2])

    env.reset()

    assert env.state.is_present(1)
    assert not env.state.is_fixed(2)


def _rejectable_env(jobs: list[int]) -> SchedulingEnv:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(len(jobs), disjunctive=False),
        [RejectableConstraint()],
        Makespan(),
        instance={"job": jobs, "processing_time": [1] * len(jobs)},
    )
    env.reset()

    return env


def test_rejecting_a_task_rejects_its_whole_job() -> None:
    env = _rejectable_env([0, 0, 1, 1])

    env.step(("reject", 1))

    assert env.state.is_absent(0)
    assert not env.state.is_fixed(2)

    env.step(("execute", 2))

    assert env.state.is_present(3)
    assert env.get_entry() == "P4|rej|C_max"


def test_rejectable_jobs_follow_the_loaded_instance() -> None:
    env = _rejectable_env([0, 0, 0])

    env.load_instance({"job": [0, 1, 2], "processing_time": [1, 1, 1]})
    env.reset()
    env.step(("reject", 0))

    assert not env.state.is_fixed(1)
    assert not env.state.is_fixed(2)


def test_groups_sharing_a_task_are_settled_together() -> None:
    env = optional_parallel_env(AllOrNothingConstraint([[0, 1], [1, 2]]), 4)

    env.step(("execute", 0))

    assert env.state.is_present(2)
    assert not env.state.is_fixed(3)


def test_a_group_joining_two_groups_merges_them() -> None:
    env = optional_parallel_env(
        AllOrNothingConstraint([[0, 1], [2, 3], [1, 2]]), 5
    )

    env.step(("execute", 0))

    assert env.state.is_present(3)
    assert not env.state.is_fixed(4)


def test_backtracking_restores_a_settled_group() -> None:
    env = optional_parallel_env(AllOrNothingConstraint([[0, 1]]), 2)

    mark = env.checkpoint()
    env.step(("reject", 0))
    env.backtrack(mark)
    env.step(("execute", 0))

    assert env.state.is_present(1)

import random

import pytest

import logging

from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.constraints import (
    PrecedenceConstraint,
    NoWaitConstraint,
    NonOverlapConstraint,
    ReleaseDateConstraint,
    DeadlineConstraint,
    ResourceConstraint,
    NonRenewableResourceConstraint,
    SetupConstraint,
    MachineBreakdownConstraint,
    MachineEligibilityConstraint,
    ConstantProcessingTime,
    PreemptionConstraint,
    OptionalityConstraint,
)
from cpscheduler.environment.schedule_setup import (
    SingleMachineSetup,
    IdenticalParallelMachineSetup,
)


def test_precedence_constraint_on_reset() -> None:
    instance = {"processing_time": [3, 2, 4, 1, 5]}
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        (PrecedenceConstraint({0: [1, 2], 1: [3, 4], 2: [4]}),),
        instance=instance,
    )

    env.reset()

    assert env.state.get_start_lb(1) == 3
    assert env.state.get_start_lb(2) == 3
    assert env.state.get_start_lb(3) == 5
    assert env.state.get_start_lb(4) == 7

def test_no_wait_constraint() -> None:
    instance = {"processing_time": [3, 2]}
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[NoWaitConstraint({0: [1]})],
        instance=instance,
    )
    env.reset()

    env.step(("execute", 0, 0))

    assert env.state.get_start_ub(1) == 3


def test_no_wait_constraint_parallel() -> None:
    instance = {"processing_time": [3, 2], "release_time": [0, 10]}
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(n_machines=2, disjunctive=False),
        constraints=[
            NoWaitConstraint({0: [1]}),
            ReleaseDateConstraint("release_time"),
        ],
        instance=instance,
    )

    env.reset()

    assert env.state.get_start_lb(1) == 10
    assert env.state.get_end_lb(0) == 10
    assert env.state.get_start_lb(0) == 7

def test_non_overlap_constraint() -> None:
    instance = {"processing_time": [3, 2, 4]}
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[NonOverlapConstraint([[0, 1], [0, 2]])],
        instance=instance,
    )

    env.reset()

    env.step(("execute", 0, 0))

    assert env.state.get_start_lb(1) == 3
    assert env.state.get_start_lb(2) == 3


def test_deadline_constraint() -> None:
    instance = {"processing_time": [2, 2]}
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[DeadlineConstraint(const_due=10)],
        instance=instance,
    )

    env.reset()

    assert env.state.get_end_ub(0) == 10
    assert env.state.get_end_ub(1) == 10


def test_resource_constraint() -> None:
    instance = {
        "processing_time": [3, 2, 7, 1],
        "resource_0": [2, 2, 1, 3],
    }
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[ResourceConstraint([3], ["resource_0"])],
        instance=instance,
    )

    env.reset()

    env.step(("execute", 0))

    assert env.state.get_start_lb(1) == 3
    assert env.state.get_start_lb(2) == 0
    assert env.state.get_start_lb(3) == 3

    env.step(("execute", 2))

    assert env.state.get_start_lb(3) == 7

def test_nonrenewable_resource_constraint() -> None:
    instance = {"processing_time": [1, 1], "resource_0": [1, 1]}
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[NonRenewableResourceConstraint([1], ["resource_0"])],
        instance=instance,
    )

    env.reset()
    env.step(("execute", 0))

    assert not env.state.is_feasible(1, 0)


def test_setup_constraint() -> None:
    instance = {"processing_time": [3, 2]}
    setup_times = {0: {1: 4}, 1: {0: 1}}
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[SetupConstraint(setup_times)],
        instance=instance,
    )

    env.reset()
    env.step(("execute", 0))

    assert env.state.get_start_lb(1) == 7

    env.reset()
    env.step(("execute", 1))

    assert env.state.get_start_lb(0) == 3



def test_machine_breakdown_constraint() -> None:
    instance = {"processing_time": [3]}
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        constraints=[MachineBreakdownConstraint({0: [(2, 5)]})],
        instance=instance,
    )

    env.reset()

    assert env.state.get_start_lb(0, 0) == 5


def test_machine_eligibility_constraint(caplog: pytest.LogCaptureFixture) -> None:
    instance = {"processing_time": [3]}
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(n_machines=2, disjunctive=False),
        constraints=[MachineEligibilityConstraint({0: [1]})],
        instance=instance,
    )

    env.reset()

    machines = list(env.state.tasks[0].machines)
    assert machines == [1]

    # Capture a logging error, expect an Error log
    # using logging module to capture logs
    with caplog.at_level(logging.ERROR):
        env.step(("execute", 0, 0, 0))  # task_id, machine_id, time

    assert "Machine 0 is not eligible for task 0." in caplog.text

def test_constant_processing_time_overrides_processing_times() -> None:
    instance = {"processing_time": [random.randint(1, 10) for _ in range(10)]}
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(n_machines=2, disjunctive=False),
        constraints=[ConstantProcessingTime(1)],
        instance=instance,
    )

    env.reset()

    for task in env.state.tasks:
        assert all(p == 1 for p in task.processing_times.values())


# def test_passive_constraints_set_task_flags() -> None:
#     instance = {"processing_time": [1, 1]}
#     env = _make_env(
#         instance,
#         SingleMachineSetup(disjunctive=False),
#         constraints=[PreemptionConstraint([0]), OptionalityConstraint()],
#     )

#     assert env.state.tasks[0].preemptive
#     assert not env.state.tasks[1].preemptive
#     assert all(task.optional for task in env.state.tasks)

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    Makespan,
    SchedulingEnv,
    SetupConstraint,
    SingleMachineSetup,
)


def test_setup_time_separates_consecutive_tasks() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [SetupConstraint({0: {1: 5}})],
        Makespan(),
        instance={"processing_time": [1, 1]},
    )
    env.reset()

    env.step([("execute", 0), ("execute", 1)])

    assert env.state.get_start(1) == 1 + 5


def test_setup_time_only_applies_in_its_direction() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [SetupConstraint({0: {1: 5}})],
        Makespan(),
        instance={"processing_time": [1, 1]},
    )
    env.reset()

    env.step([("execute", 1), ("execute", 0)])

    assert env.state.get_start(0) == 1


def test_setup_time_ignores_tasks_in_other_machines() -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [SetupConstraint({0: {1: 5}})],
        Makespan(),
        instance={"processing_time": [1, 1]},
    )
    env.reset()

    env.step([("execute", 0, 0), ("execute", 1, 1)])

    assert env.state.get_start(1) == 0

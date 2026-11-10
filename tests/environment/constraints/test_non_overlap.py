from cpscheduler.environment import (
    Makespan,
    NonOverlapConstraint,
    SchedulingEnv,
    SingleMachineSetup,
)


def _two_episodes(constraint: NonOverlapConstraint) -> list[int]:
    env = SchedulingEnv(
        SingleMachineSetup(disjunctive=False),
        [constraint],
        Makespan(),
        instance={"processing_time": [2, 3]},
    )

    starts = []
    for _ in range(2):
        env.reset()
        env.step([("execute", 0), ("execute", 1)])
        starts.append(env.state.get_start(1))

    return starts


def test_added_groups_hold_in_every_episode() -> None:
    constraint = NonOverlapConstraint()
    constraint.add_group([0, 1])

    assert _two_episodes(constraint) == [2, 2]


def test_tasks_added_to_a_group_hold_in_every_episode() -> None:
    constraint = NonOverlapConstraint()
    constraint.add_task(0, 0)
    constraint.add_task(0, 1)

    assert _two_episodes(constraint) == [2, 2]


def test_removed_groups_stay_removed() -> None:
    constraint = NonOverlapConstraint(task_groups=[[0, 1]])
    constraint.remove_group(0)

    assert _two_episodes(constraint) == [0, 0]

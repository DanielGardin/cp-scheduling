import pytest

from cpscheduler.environment import JobShopSetup, Makespan, SchedulingEnv


def test_job_shop_requires_contiguous_operations() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        SchedulingEnv(
            JobShopSetup(),
            objective=Makespan(),
            instance={
                "job": [0, 0],
                "operation": [0, 2],
                "machine": [0, 1],
                "processing_time": [1, 1],
            },
        )


def test_job_ids_with_gaps_are_accepted() -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        instance={
            "job": [3, 3],
            "operation": [0, 1],
            "machine": [0, 1],
            "processing_time": [1, 1],
        },
    )
    env.reset()

    assert env.get_action_support() == [0]

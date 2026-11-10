import pytest

from cpscheduler.environment import (
    JobShopSetup,
    Makespan,
    OpenShopSetup,
    SchedulingEnv,
    SingleMachineSetup,
)
from cpscheduler.environment.constants import MAX_TIME


@pytest.mark.parametrize(
    ("duration", "message"),
    [(-1, "negative"), (MAX_TIME, "too large"), (MAX_TIME + 1, "too large")],
)
def test_processing_time_out_of_range_is_rejected(
    duration: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SchedulingEnv(
            SingleMachineSetup(),
            objective=Makespan(),
            instance={"processing_time": [duration]},
        )


def test_reloading_another_job_structure_leaves_nothing_behind() -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        instance={
            "job": [0, 0, 0],
            "operation": [0, 1, 2],
            "machine": [0, 1, 0],
            "processing_time": [1, 1, 1],
        },
    )
    env.reset()

    assert env.get_action_support() == [0]

    env.load_instance(
        {
            "job": [0, 1, 2],
            "operation": [0, 0, 0],
            "machine": [0, 1, 0],
            "processing_time": [1, 1, 1],
        }
    )
    env.reset()

    assert env.get_action_support() == [0, 1, 2]


def test_reloading_an_open_shop_keeps_no_job_groups() -> None:
    env = SchedulingEnv(
        OpenShopSetup(),
        objective=Makespan(),
        instance={
            "job": [0, 0],
            "machine": [0, 1],
            "processing_time": [2, 2],
        },
    )
    env.load_instance(
        {"job": [0, 1], "machine": [0, 1], "processing_time": [2, 2]}
    )
    env.reset()

    # Separate jobs on separate machines can run together
    env.step([("execute", 0), ("execute", 1)])

    assert env.state.get_start(1) == 0

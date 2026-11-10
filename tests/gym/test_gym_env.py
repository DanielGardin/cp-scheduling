import pytest

gym = pytest.importorskip("gymnasium")
from cpscheduler.environment import (  # noqa: E402
    FlowShopSetup,
    JobShopSetup,
    Makespan,
    NonOverlapConstraint,
    OpenShopSetup,
    SingleMachineSetup,
)
from cpscheduler.environment.setups import ScheduleSetup  # noqa: E402
from cpscheduler.gym import SchedulingEnvGym  # noqa: E402


@pytest.mark.parametrize(
    ("setup", "instance"),
    [
        pytest.param(
            FlowShopSetup(),
            {"job": [0, 0], "operation": [0, 1], "processing_time": [1, 2]},
            id="flow_shop",
        ),
        pytest.param(
            OpenShopSetup(),
            {"job": [0, 0], "machine": [0, 1], "processing_time": [1, 2]},
            id="open_shop",
        ),
    ],
)
def test_observations_belong_to_the_observation_space(
    setup: ScheduleSetup, instance: dict[str, list[int]]
) -> None:
    env = SchedulingEnvGym(setup, objective=Makespan(), instance=instance)

    obs, _ = env.reset()

    assert env.observation_space.contains(obs)

    obs, *_ = env.step(("execute", 0))

    assert env.observation_space.contains(obs)


def test_groups_can_be_loaded_after_the_environment_is_created() -> None:
    env = SchedulingEnvGym(
        SingleMachineSetup(disjunctive=False),
        [NonOverlapConstraint()],
        Makespan(),
    )

    obs, _ = env.reset(
        options={
            "instance": {
                "processing_time": [2, 2],
                "non_overlap_groups": [[0, 1]],
            }
        }
    )

    assert env.observation_space.contains(obs)

    env.step([("execute", 0), ("execute", 1)])

    assert env.core.state.get_start(1) == 2


def test_render_mode_reaches_the_renderer() -> None:
    env = SchedulingEnvGym(
        JobShopSetup(),
        objective=Makespan(),
        instance={
            "job": [0, 0, 1, 1],
            "operation": [0, 1, 0, 1],
            "machine": [0, 1, 1, 0],
            "processing_time": [3, 2, 2, 4],
        },
        render_mode="ansi",
    )
    env.reset()

    env.step(("execute", 0))

    assert env.render_mode == "ansi"
    assert env.render()

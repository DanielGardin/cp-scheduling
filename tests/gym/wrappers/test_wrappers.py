from typing import Literal

import pytest

gym = pytest.importorskip("gymnasium")
from cpscheduler.environment import (  # noqa: E402
    JobShopSetup,
    Makespan,
    ReleaseDateConstraint,
    SingleMachineSetup,
)
from cpscheduler.environment.observation.default import (  # noqa: E402
    DefaultObsType,
)
from cpscheduler.gym import SchedulingEnvGym  # noqa: E402
from cpscheduler.gym.wrappers import (  # noqa: E402
    ForcedActionWrapper,
    PermutationActionWrapper,
    SingleActionNoopWrapper,
    SingleActionWrapper,
    SingleAssignmentWrapper,
)

# The optimal makespan is 7, a serial schedule of 0, 1, 2, 3 ends at 11
JOB_SHOP: dict[str, list[int]] = {
    "job": [0, 0, 1, 1],
    "operation": [0, 1, 0, 1],
    "machine": [0, 1, 1, 0],
    "processing_time": [3, 2, 2, 4],
}


def _job_shop() -> SchedulingEnvGym:
    return SchedulingEnvGym(
        JobShopSetup(), objective=Makespan(), instance=JOB_SHOP
    )


@pytest.mark.parametrize(
    "wrapper", [SingleActionWrapper, SingleActionNoopWrapper]
)
def test_task_masks_lead_to_a_complete_schedule(
    wrapper: type[SingleActionWrapper[DefaultObsType]]
    | type[SingleActionNoopWrapper[DefaultObsType]],
) -> None:
    env = wrapper(_job_shop())
    env.reset(seed=0)

    steps = 0
    terminated = False
    while not terminated:
        mask = env.action_masks()

        assert mask is not None
        assert mask[:4].any()

        *_, terminated, truncated, info = env.step(int(mask.argmax()))

        assert not truncated

        steps += 1

    assert steps == 4
    assert info["objective_value"] == 7


def test_assignment_mask_leads_to_a_complete_schedule() -> None:
    env = SingleAssignmentWrapper(_job_shop())
    env.reset(seed=0)

    terminated = False
    while not terminated:
        mask = env.action_masks()

        assert mask is not None

        task, machine = (
            int(i) for i in next(zip(*mask.nonzero(), strict=True))
        )
        *_, terminated, truncated, info = env.step([task, machine])

        assert not truncated

    assert info["objective_value"] == 7


@pytest.mark.parametrize(
    ("generation", "expected"), [("serial", 11), ("parallel", 7)]
)
def test_permutation_follows_the_generation_scheme(
    generation: Literal["serial", "parallel"], expected: int
) -> None:
    env = PermutationActionWrapper(_job_shop(), schedule_generation=generation)
    env.reset()

    *_, terminated, _, info = env.step([0, 1, 2, 3])

    assert terminated
    assert info["objective_value"] == expected


def test_mask_has_options_on_reset_with_late_releases() -> None:
    env = SingleActionWrapper(
        SchedulingEnvGym(
            SingleMachineSetup(),
            [ReleaseDateConstraint()],
            Makespan(),
            instance={"processing_time": [1, 4], "release_time": [5, 5]},
        )
    )
    env.reset()

    mask = env.action_masks()

    assert mask is not None
    assert mask.tolist() == [1, 1]


def test_action_space_follows_the_loaded_instance() -> None:
    env = SingleActionWrapper(
        SchedulingEnvGym(JobShopSetup(), objective=Makespan())
    )

    env.reset(options={"instance": JOB_SHOP})

    assert env.action_space.contains(3)
    assert not env.action_space.contains(4)


def test_forced_actions_are_played_without_asking() -> None:
    env = ForcedActionWrapper(_job_shop())
    env.reset()

    # Task 2 becomes the only option after task 0
    env.step(("execute", 0))

    assert env.get_wrapper_attr("core").state.is_fixed(2)

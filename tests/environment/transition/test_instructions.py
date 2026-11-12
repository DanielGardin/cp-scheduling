import pytest

from cpscheduler.environment import Makespan, SchedulingEnv, SingleMachineSetup
from tests.schedule_checks import TRANSITIONS


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_repeating_a_fixed_task_is_rejected(transition: str) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        transition=transition,
        instance={"processing_time": [1, 1]},
    )
    env.reset()

    env.step(("execute", 0))

    with pytest.raises(ValueError, match="already fixed"):
        env.step(("execute", 0))

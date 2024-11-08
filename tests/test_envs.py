from pathlib import Path

import pytest

import numpy as np

from cpscheduler.environment import SchedulingCPEnv, PrecedenceConstraint, NonOverlapConstraint, Makespan
from cpscheduler.utils import read_instance


TEST_INSTANCES = [
    "dmu04",
    "la10",
    "orb01",
    "swv12",
    "ta20",
    "tai_j10_m10_1"
]

@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_env(instance_name: str) -> None:
    path = Path(__file__).parent.parent / f"instances/{instance_name}.txt"

    instance, _ = read_instance(path)

    env = SchedulingCPEnv(instance, "processing_time")

    env.add_constraint(
        PrecedenceConstraint.jobshop_precedence(env.tasks, 'job', 'operation')
    )

    env.add_constraint(
        NonOverlapConstraint.jobshop_non_overlap(env.tasks, 'machine')
    )

    env.add_objective(
        Makespan(env.tasks)
    )

    env.reset()

    assert env.current_time == 0

    first_action = np.array([0])
    obs, reward, terminated, truncated, info = env.step(first_action, time_skip=0)

    last_objective_values = env.get_objective_values()

    assert reward == 0
    assert not terminated
    assert not truncated
    assert info['current_time'] == 0

    new_obs, new_reward, new_terminated, new_truncated, new_info = env.step(time_skip=100)

    assert new_reward == last_objective_values - env.get_objective_values()
    assert not new_terminated
    assert not new_truncated
    assert new_info['current_time'] == min(100, env.tasks.durations[0])


@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_not_enforce_order(instance_name: str) -> None:
    path = Path(__file__).parent.parent / f"instances/{instance_name}.txt"

    instance, _ = read_instance(path)

    env = SchedulingCPEnv(instance, "processing_time")

    env.add_constraint(
        PrecedenceConstraint.jobshop_precedence(env.tasks, 'job', 'operation')
    )

    env.add_constraint(
        NonOverlapConstraint.jobshop_non_overlap(env.tasks, 'machine')
    )

    env.add_objective(
        Makespan(env.tasks)
    )

    obs, info = env.reset()
    spt = np.argsort(obs['processing_time'])

    obs, reward, terminated, truncated, info = env.step(spt, time_skip=None, enforce_order=False)

    assert (obs['buffer'] == 'finished').all()
    assert terminated

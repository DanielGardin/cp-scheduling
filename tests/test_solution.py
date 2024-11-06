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
    "ta07",
    "tai_j10_m10_1"
]

@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_cp_solution(instance_name: str) -> None:
    path = Path(__file__).parent.parent / f"instances/{instance_name}.txt"

    instance = read_instance(path)

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

    order, objective_value, is_optimal = env.get_cp_solution(timelimit=1)

    obs, reward, terminated, truncated, info = env.step(order, time_skip=None)

    assert (obs['buffer'] == 'finished').all()
    assert (obs['remaining_time'] == 0).all()
    assert reward < 0
    assert terminated
    assert info['current_time'] == objective_value



@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_partial_cp_solution(instance_name: str) -> None:
    path = Path(__file__).parent.parent / f"instances/{instance_name}.txt"

    instance = read_instance(path)

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

    time_skip = 500

    obs, reward, terminated, truncated, info = env.step(spt, time_skip=time_skip, enforce_order=False)

    assert info['current_time'] == time_skip
    assert not terminated

    order, objective_value, is_optimal = env.get_cp_solution(timelimit=1)

    assert (np.sort(order) == env.tasks.ids[obs['buffer'] == 'awaiting']).all()

    obs, reward, terminated, truncated, info = env.step(order, time_skip=None)

    assert (obs['buffer'] == 'finished').all()
    assert (obs['remaining_time'] == 0).all()
    assert reward < 0
    assert terminated
    assert info['current_time'] == objective_value
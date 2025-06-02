from typing import Literal
from pathlib import Path

import pytest

from cpscheduler.environment import read_jsp_instance
from cpscheduler.common_envs import JobShopEnv


TEST_INSTANCES = [
    "dmu04",
    "la10",
    "orb01",
    "swv12",
    # "ta80",
]

SOLVERS = [
    "cplex",
    "ortools",
]

@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
@pytest.mark.parametrize("cp_solver", SOLVERS)
def test_cp_solution(instance_name: str, cp_solver: Literal['cplex', 'ortools']) -> None:
    path = Path(__file__).parent.parent / f"instances/jobshop/{instance_name}.txt"

    instance, _ = read_jsp_instance(path)

    env = JobShopEnv(instance)

    env.reset()

    order, starts, objective_value, is_optimal = env.get_cp_solution(timelimit=2, solver=cp_solver)

    obs, reward, terminated, truncated, info = env.step(order, time_skip=None)

    assert all([buffer == 'finished' for buffer in obs['buffer']])
    assert terminated
    # OR-tools do not guarantee a compressed solution, allowing for a simulated time
    # smaller than the computed objective value.
    assert info['current_time'] <= objective_value

# TODO: la10 breaks the test with an Index error
@pytest.mark.env
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_partial_cp_solution(instance_name: str) -> None:
    path = Path(__file__).parent.parent / f"instances/jobshop/{instance_name}.txt"

    instance, _ = read_jsp_instance(path)

    env = JobShopEnv(instance)

    obs, info = env.reset()
    spt = sorted(obs['processing_time'], reverse=True)

    time_skip = 500

    # This is infinite looping
    obs, reward, terminated, truncated, info = env.step(spt, time_skip=time_skip, enforce_order=False)

    assert info['current_time'] == time_skip
    assert not terminated

    order, starts, objective_value, is_optimal = env.get_cp_solution(timelimit=2)
    next_obs, next_reward, next_terminated, next_truncated, next_info = env.step(order, time_skip=None)

    assert objective_value == -(reward + next_reward)
    assert next_info['current_time'] == objective_value
    assert next_terminated
    assert all([buffer == 'finished' for buffer in next_obs['buffer']])
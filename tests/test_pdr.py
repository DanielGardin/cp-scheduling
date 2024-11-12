from pathlib import Path

import pytest

import numpy as np

from cpscheduler.common_envs import JobShopEnv
from cpscheduler.environment import read_instance

from cpscheduler.policies.heuristics import ShortestProcessingTime, MostOperationsRemaining, MostWorkRemaining

pdr_expected_results = {
    # 15x15
    5:  [1618, 1448, 1494],
    10: [1697, 1582, 1534],
    # 20x15
    15: [1835, 1682, 1696],
    20: [1710, 1622, 1689],
    # 20x20 
    25: [1950, 1957, 1941],
    30: [1999, 2017, 1935],
    # 30x15 
    35: [2497, 2255, 2226],
    40: [2301, 2028, 2205],
    # 30x20
    45: [2640, 2487, 2524],
    50: [2429, 2469, 2493],
    # 50x15
    60: [3500, 3044, 3122],
    # 50x20
    70: [3801, 3590, 3482],
    # 100x20
    80: [5848, 5707, 5505],
}

heuristics = {
    "SPT"  : ShortestProcessingTime(),
    "MOPNR": MostOperationsRemaining(),
    "MWKR" : MostWorkRemaining()
}

@pytest.mark.parametrize("instance_no", pdr_expected_results)
def test_pdr(instance_no: int) -> None:
    path = Path(__file__).parent.parent / f"instances/jobshop/ta{instance_no:02d}.txt"

    instance, _ = read_instance(path)

    env = JobShopEnv(instance, dataframe_obs=False)

    result = {}

    for name, heuristic in heuristics.items():
        obs, info = env.reset()

        action = heuristic(obs)
        obs, reward, terminated, truncated, info = env.step(action, enforce_order=False)

        result[name] = info['current_time']

    assert result == {
        name: pdr_expected_results[instance_no][i] for i, name in enumerate(heuristics)
    }
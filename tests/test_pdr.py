import pytest

from common import env_setup

from cpscheduler.policies.heuristics import ShortestProcessingTime, MostOperationsRemaining, MostWorkRemaining, PriorityDispatchingRule

pdr_expected_results = {
    # 15x15
    1:  [1462, 1438, 1491],
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

heuristics: dict[str, PriorityDispatchingRule] = {
    "SPT"  : ShortestProcessingTime(),
    "MOPNR": MostOperationsRemaining(),
    "MWKR" : MostWorkRemaining()
}

@pytest.mark.heuristics
@pytest.mark.parametrize("instance_no", pdr_expected_results)
def test_pdr(instance_no: int) -> None:
    env = env_setup(f"ta{instance_no:02d}")

    result = {}

    for name, heuristic in heuristics.items():
        obs, info = env.reset()

        action = heuristic(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        result[name] = info['current_time']
        assert terminated

    assert result == {
        name: pdr_expected_results[instance_no][i] for i, name in enumerate(heuristics)
    }
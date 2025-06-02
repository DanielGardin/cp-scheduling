import pytest

from common import env_setup

from cpscheduler.heuristics import ShortestProcessingTime, MostOperationsRemaining, MostWorkRemaining, PriorityDispatchingRule

pdr_expected_results = {
    # 15x15
    "ta01":  [1462, 1438, 1491],
    "ta05":  [1618, 1448, 1494],
    "ta10": [1697, 1582, 1534],
    # 20x15
    "ta15": [1835, 1682, 1696],
    "ta20": [1710, 1622, 1689],
    # 20x20 
    "ta25": [1950, 1957, 1941],
    "ta30": [1999, 2017, 1935],
    # 30x15 
    "ta35": [2497, 2255, 2226],
    "ta40": [2301, 2028, 2205],
    # 30x20
    "ta45": [2640, 2487, 2524],
    "ta50": [2429, 2469, 2493],
    # 50x15
    "ta60": [3500, 3044, 3122],
    # 50x20
    "ta70": [3801, 3590, 3482],
    # 100x20
    "ta80": [5848, 5707, 5505],
}

heuristics: dict[str, PriorityDispatchingRule] = {
    "SPT"  : ShortestProcessingTime(),
    "MOPNR": MostOperationsRemaining(),
    "MWKR" : MostWorkRemaining()
}

@pytest.mark.heuristics
@pytest.mark.parametrize("instance_name", pdr_expected_results)
def test_pdr(instance_name: str) -> None:
    env = env_setup(instance_name)

    result = {}

    for name, heuristic in heuristics.items():
        obs, info = env.reset()

        action = heuristic(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        result[name] = info['current_time']
        assert terminated

    assert result == {
        name: pdr_expected_results[instance_name][i] for i, name in enumerate(heuristics)
    }
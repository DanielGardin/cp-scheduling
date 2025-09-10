import pytest

from common import env_setup

from cpscheduler.heuristics import (
    ShortestProcessingTime,
    MostOperationsRemaining,
    MostWorkRemaining,
    PriorityDispatchingRule,
)

pdr_expected_results = {
    "ta01": {
        "SPT": 1462,
        "MOPNR": 1438,
        "MWKR": 1491,
    },
    "ta05": {
        "SPT": 1618,
        "MOPNR": 1448,
        "MWKR": 1494,
    },
    "ta10": {"SPT": 1697, "MOPNR": 1582, "MWKR": 1534},
    "ta15": {"SPT": 1835, "MOPNR": 1682, "MWKR": 1696},
    "ta20": {"SPT": 1710, "MOPNR": 1622, "MWKR": 1689},
    "ta25": {"SPT": 1950, "MOPNR": 1957, "MWKR": 1941},
    "ta30": {"SPT": 1999, "MOPNR": 2017, "MWKR": 1935},
    "ta35": {"SPT": 2497, "MOPNR": 2255, "MWKR": 2226},
    "ta40": {"SPT": 2301, "MOPNR": 2028, "MWKR": 2205},
    "ta45": {"SPT": 2640, "MOPNR": 2487, "MWKR": 2524},
    "ta50": {"SPT": 2429, "MOPNR": 2469, "MWKR": 2493},
    "ta60": {"SPT": 3500, "MOPNR": 3044, "MWKR": 3122},
    "ta70": {"SPT": 3801, "MOPNR": 3590, "MWKR": 3482},
    "ta80": {"SPT": 5848, "MOPNR": 5707, "MWKR": 5505},
}

heuristics: dict[str, PriorityDispatchingRule] = {
    "SPT": ShortestProcessingTime(),
    "MOPNR": MostOperationsRemaining(),
    "MWKR": MostWorkRemaining(),
}


@pytest.mark.heuristics
@pytest.mark.parametrize("instance_name", pdr_expected_results)
@pytest.mark.parametrize("heuristic", heuristics)
def test_pdr(instance_name: str, heuristic: str) -> None:
    env = env_setup(instance_name)

    obs, info = env.reset()

    action = heuristics[heuristic](obs)
    obs, reward, terminated, truncated, info = env.step(action)

    assert info["current_time"] == pdr_expected_results[instance_name][heuristic]
    assert terminated

@pytest.mark.heuristics
@pytest.mark.parametrize("instance_name", pdr_expected_results)
def test_dynamic(instance_name: str) -> None:
    env = env_setup(instance_name)

    obs, info = env.reset()

    action = ShortestProcessingTime()(obs)
    *_, info = env.step(action)

    static_time = info["current_time"]

    obs, info = env.reset()

    done = False
    dynamic_actor = ShortestProcessingTime(available=True)
    while not done:
        single_action = dynamic_actor(obs)[0]
        obs, _, done, _, info = env.step(single_action)

    assert info["current_time"] == static_time
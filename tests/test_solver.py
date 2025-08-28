import pytest

from common import env_setup, TEST_INSTANCES

@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_pause(instance_name: str) -> None:
    pytest.importorskip("pulp")
    from cpscheduler.solver import PulpSolver

    env = env_setup("la01")

    solver = PulpSolver(env)
    action, objective, optimal = solver.solve(
        solver_tag="CPLEX_CMD",
        time_limit=5
    )

    env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

    assert reward - (-objective) < 1e-5

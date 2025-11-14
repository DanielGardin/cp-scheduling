import pytest

from time import process_time

from common import env_setup, TEST_INSTANCES

import logging

logger = logging.getLogger(__name__)

@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
def test_solve(instance_name: str) -> None:
    pytest.importorskip("pulp")
    from cpscheduler.solver import PulpSolver

    env = env_setup(instance_name)
    env.reset()

    solver = PulpSolver(env)
    # Measure time to solve
    
    time = -process_time()
    action, objective, optimal = solver.solve(
        solver_tag="CPLEX_CMD",
        time_limit=5,
        keep_files=False,
    )
    time += process_time()
    logger.info(f"Solved instance {instance_name} in {time:.2f}s")

    assert len(action) == env.state.n_tasks

    env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

    logger.info(
        f"Tested instance {instance_name}: {objective=} {optimal=} {reward=}"
    )

    assert reward - (-objective) < 1e-5

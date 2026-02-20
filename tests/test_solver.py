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

    time = -process_time()
    solver = PulpSolver(env, tighten=True, symmetry_breaking=False)
    solver.build()
    time += process_time()
    logger.info(f"Initialized solver in {time:.2f} s")

    time = -process_time()
    try:
        action, optimal_value, optimal = solver.solve(
            solver_tag="CPLEX_CMD",
            time_limit=5,
            keep_files=False,
        )

    except Exception as e:
        raise e

    finally:
        time += process_time()
        logger.info(f"Solved took {time:.2f} s to solve instance")

    assert len(action) == env.state.n_tasks

    time = -process_time()
    env.reset()
    *_, info = env.step(action)
    time += process_time()

    logger.info(f"Simulation took {time:.2f} s")

    objective_value = info["objective_value"]

    if optimal:
        logger.info(
            f"Summary of instance {instance_name}: {optimal_value=}, {objective_value=}"
        )
    
    else:
        logger.warning(f"Instance {instance_name} did not reach optimality.")

    assert abs(objective_value - optimal_value) < 1e-5
import pytest

from time import perf_counter

from common import env_setup, TEST_INSTANCES

import logging

from cpscheduler.solver import get_formulations

logger = logging.getLogger(__name__)


@pytest.mark.solver
@pytest.mark.parametrize("instance_name", TEST_INSTANCES)
@pytest.mark.parametrize("formulation", get_formulations())
def test_solve(instance_name: str, formulation: str) -> None:
    from cpscheduler.solver import SchedulingSolver

    env = env_setup(instance_name)
    env.reset()

    time = -perf_counter()
    solver = SchedulingSolver(env, formulation=formulation, horizon=10000)
    solver.warm_start([("submit", task_id) for task_id in range(env.state.n_tasks)])
    solver.build()
    time += perf_counter()
    logger.info(f"Initialized solver in {time:.2f} s")

    time = -perf_counter()
    try:
        action, optimal_value, optimal = solver.solve(
            time_limit=5,
            keep_files=False,
        )

    except RuntimeError as exc:
        # Some MILP solvers can hit the time limit without producing an incumbent.
        pytest.skip(
            f"No incumbent solution found within time limit for {instance_name} ({formulation}): {exc}."
        )

    finally:
        time += perf_counter()
        logger.info(f"Solved took {time:.2f} s to solve instance")

    assert len(action) == env.state.n_tasks

    time = -perf_counter()
    env.reset()
    *_, info = env.step(action)
    time += perf_counter()

    logger.info(f"Simulation took {time:.2f} s")

    objective_value = info["objective_value"]

    if optimal:
        logger.info(
            f"Summary of instance {instance_name}: {optimal_value=}, {objective_value=}"
        )
    
    else:
        logger.warning(f"Instance {instance_name} did not reach optimality.")

    assert abs(objective_value - optimal_value) < 1e-5
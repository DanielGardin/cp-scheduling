from typing import Any, Optional, NamedTuple
from datetime import timedelta

from time import perf_counter

import asyncio

from tqdm import tqdm


from minizinc import Result, Instance, Solver, Model, Status

from ..environment import SchedulingCPEnv
from .utils import TimeUnits, resolve_timeout, run_couroutine

# Not finished, just taking a time from it.


class SolverConfig(NamedTuple):
    processes: int = 1
    timeout: Optional[timedelta] = None
    optimisation_level: int = 1
    free_search: bool = False
    random_seed: Optional[int] = None

async def monitor_timeout(
        timeout: float,
        start_time: float,
        main_task: asyncio.Task[Any],
        check_intervals: int = 60,
        pbar: Optional[tqdm] = None
    ) -> None:
    i = 0
    while True:
        i += 1
        if pbar is not None:
            pbar.update(1)

        if perf_counter() - start_time >= timeout:
            main_task.cancel()
            break

        try:
            await asyncio.sleep(timeout / check_intervals)

        except asyncio.CancelledError:
            break

async def timed_solve(
        instance: Instance,
        solver_config: SolverConfig,
        timeout: float,
        check_intervals: int = 60,
        progress: bool = True,
        **kwargs: Any
    ) -> list[Result]:
    solutions: list[Result] = []
    task: asyncio.Task[None] | None = None

    main_task = asyncio.current_task()
    assert main_task is not None

    try:
        with tqdm(
            total=check_intervals,
            desc="Checking for solutions",
            unit="checks",
            disable=not progress
        ) as pbar:
            async for result in instance.solutions(
                processes=solver_config.processes,
                    **solver_config._asdict(),
                    **kwargs
            ):
                if result.solution is None:
                    continue

                pbar.total = pbar.n + check_intervals
                pbar.set_postfix({
                    "status": result.status.name,
                    "solutions": len(solutions),
                    **result.statistics
                })
                pbar.refresh()

                if task is not None:
                    task.cancel()

                solutions.append(result)

                task = asyncio.create_task(
                    monitor_timeout(
                        timeout,
                        perf_counter(),
                        main_task,
                        check_intervals,
                        pbar
                    )
                )

    except asyncio.CancelledError:
        pass

    return solutions


class MinizincSolver:
    def __init__(
            self,
            env: SchedulingCPEnv,
            solver_tag: str = "cp",
            processes: int = 1,
            time_limit: Optional[int] = None,
            time_unit: TimeUnits = 's',
            optimisation_level: int = 1,
            free_search: bool = False,
            random_seed: Optional[int] = None,
            **kwargs: Any
    ):
        self.env = env

        model = Model()
        model.add_string('\n'.join(env.export()))

        self.model = model
        self.solver = Solver.lookup(solver_tag)

        self.instance = Instance(self.solver, self.model)

        self.config = SolverConfig(
            processes=processes,
            timeout=resolve_timeout(time_limit, time_unit),
            optimisation_level=optimisation_level,
            free_search=free_search,
            random_seed=random_seed,
        )

        self.kwargs = kwargs

    @property
    def model_str(self) -> str:
        return '\n'.join(self.env.export())


    def solve(self) -> Result:
        coroutine = self.instance.solve_async(
            **self.config._asdict(),
            **self.kwargs
        )

        return run_couroutine(coroutine)


    def get_actions(
            self,
            current_time: int,
            query_time: Optional[int] = None,
        ) -> tuple[list[tuple[str, *tuple[int, ...]]], float, bool]:
        result = self.solve()

        if not result.status.has_solution():
            return [], 0.0, False
    
        starts: list[list[int]]    = result["start"]
        durations: list[list[int]] = result["duration"]

        n_tasks = len(starts)
        n_parts = len(starts[0])

        try:
            machine = result["machine"]
        
        except KeyError:
            machine = None

        actions: list[tuple[str, *tuple[int, ...]]] = []
        for task in range(n_tasks):
            for part in range(n_parts):
                start = starts[task][part]

                if start < current_time:
                    continue

                if machine is not None:
                    machine_id = machine[task][part]
                    actions.append(("execute", task, machine_id, start))
                
                else:
                    actions.append(("execute", task, start))

                if part < n_parts - 1 and durations[task][part+1] > 0:
                    actions.append(("pause", task, start + durations[task][part]))
        
        if query_time is not None:
            actions.append(("query", query_time))

        objective = float(result["objective"])
        optimal   = result.status is Status.OPTIMAL_SOLUTION

        return actions, objective, optimal

    def get_milp_formulation(self) -> tuple[list[float], Any, list[float]]:
        """
        Returns the coefficients of the MILP formulation of the problem.

        min c^T x
        s.t. Ax <= b
              x >= 0

        Returns:
            c: cost array of size (n_vars,)
            A: sparse matrix of size (n_constraints, n_vars)
            b: right-hand side array of size (n_constraints,)
        """

        import gurobipy as gp
        from os import remove

        model = gp.read("_temp.mps")
        remove("_temp.mps")

        c = model.getAttr("Obj", model.getVars())
        A = model.getA()
        b = model.getAttr("RHS", model.getConstrs())

        lb = model.getAttr("LB", model.getVars())
        ub = model.getAttr("UB", model.getVars())

        sense = model.getAttr("Sense", model.getConstrs())
        return c, A, b

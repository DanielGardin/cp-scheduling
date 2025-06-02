from typing import Optional, Literal

from datetime import timedelta

from minizinc import Result, Instance, Solver, Model, Status

from .env import SchedulingCPEnv

TimeUnits = Literal['s', 'm', 'h', 'd']

def resolve_timeout(
    timeout: Optional[int],
    timeout_unit: TimeUnits,
) -> timedelta | None:
    if timeout is None:
        return None

    if timeout_unit == "s":
        return timedelta(seconds=timeout)
    
    if timeout_unit == "m":
        return timedelta(minutes=timeout)
    
    if timeout_unit == "h":
        return timedelta(hours=timeout)
    
    if timeout_unit == "d":
        return timedelta(days=timeout)
    
    raise ValueError(f"Time unit {timeout_unit} not recognized. Use 's', 'm', 'h', or 'd'.")


class MinizincSolver:
    def __init__(
            self,
            env: SchedulingCPEnv,
            solver_tag: str = "cp",
            timeout: Optional[int] = None,
            timeout_unit: TimeUnits = 's',
            n_processors: Optional[int] = None,
            optimisation_level: int = 1,
            *,
            free_search: bool = False,
            random_seed: Optional[int] = None,
    ):
        self.env = env
        self.solver_tag = solver_tag
        self.timeout = resolve_timeout(timeout, timeout_unit)
        self.n_processors = n_processors
        self.optimisation_level = optimisation_level
        self.free_search = free_search
        self.random_seed = random_seed


    @property
    def model_str(self) -> str:
        return '\n'.join(self.env.export())

    async def solve_async(
            self,
            instance: Instance,
    ) -> Result:
        return await instance.solve_async(
            n_processors=self.n_processors,
            timeout=self.timeout,
            optimisation_level=self.optimisation_level,
            free_search=self.free_search,
            random_seed=self.random_seed,
        )


    def solve(self) -> Result:
        model = Model()
        model.add_string(self.model_str)

        solver = Solver.lookup(self.solver_tag)

        instance = Instance(solver, model)

        try:
            result = instance.solve(
                timeout=self.timeout,
                n_processors=self.n_processors,
                optimisation_level=self.optimisation_level,
                free_search=self.free_search,
                random_seed=self.random_seed,
            )

        except RuntimeError:
            import asyncio
            loop = asyncio.get_event_loop()

            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()

            result = loop.run_until_complete(self.solve_async(instance))

        return result


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
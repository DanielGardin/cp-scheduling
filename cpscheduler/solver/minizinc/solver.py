from typing import Any
from collections.abc import Sequence
from typing_extensions import Unpack

from datetime import timedelta

from time import perf_counter

import asyncio

from tqdm import tqdm

from minizinc import (
    Result,
    Instance,
    Solver,
    Model,
    Status,
    Driver,
    default_driver,
)

from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.des import SingleAction, ActionType
from cpscheduler.common import unwrap_env


from cpscheduler.solver.cp.utils import (
    TimeUnits,
    resolve_timeout,
    run_couroutine,
)

from cpscheduler.solver.minizinc.minizinc_utils import SolverConfig


class MinizincSolver:
    def __init__(self, env: Any | SchedulingEnv):
        """
        Initialize the solver using MiniZinc.

        Attributes:
            env: SchedulingEnv
                The environment containing the scheduling problem.
                It must be loaded with an instance before initializing the solver.

        """
        self.env = unwrap_env(env)

        if not self.env.state.loaded:
            raise ValueError(
                "Environment must be loaded before initializing the solver."
            )

        self.solver: Solver | None = None
        self.config: SolverConfig = {}
        self.model = self.build_model(self.env)

    @classmethod
    def available_solvers(cls) -> list[str]:
        """
        Get a list of available MiniZinc solvers.

        Returns:
            list[str]: A list of available MiniZinc solver names.
        """
        driver = default_driver

        if driver is None:
            return []

        available_solvers: dict[str, Solver] = {}
        for tag, solvers in driver.available_solvers().items():
            for solver in solvers:
                if solver.name not in available_solvers and solver.executable:
                    available_solvers[solver.name] = solver

        return list(available_solvers.keys())

    def set_solver(
        self,
        solver_tag: str,
        **solver_kwargs: Unpack[SolverConfig],
    ) -> None:
        self.solver = Solver.lookup(solver_tag)
        self.config = self.config | solver_kwargs

    @staticmethod
    def build_model(
        env: SchedulingEnv,
    ) -> Model:
        model = Model()

        return model

    def warm_start(self, action: ActionType) -> None:
        pass

    def solve(
        self,
        solver_tag: str | None = None,
        **solver_kwargs: Unpack[SolverConfig],
    ) -> tuple[Sequence[SingleAction], float, int]:
        if solver_tag is not None:
            self.set_solver(solver_tag)

        if self.solver is None:
            raise ValueError(
                "Solver not set. Please set a solver before solving."
            )

        instance = Instance(self.solver, self.model)

        result = run_couroutine(
            instance.solve_async(**(self.config | solver_kwargs))
        )

        if not result.status.has_solution():
            return [], 0.0, result.status.value

        ...

        return [], 0.0, 0


if __name__ == "__main__":
    print("Available MiniZinc solvers:", MinizincSolver.available_solvers())

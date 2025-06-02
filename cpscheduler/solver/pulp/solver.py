from typing import Any

from pulp import LpProblem, LpSolver, LpMinimize, LpMaximize
import pulp as pl

from cpscheduler.environment import SchedulingEnv

from .tasks import PulpVariables
from .setup import export_setup_pulp
from .constraint import export_constraint_pulp
from .objective import export_objective_pulp


class PulpSolver:
    solver: LpSolver

    def __init__(
            self,
            env: SchedulingEnv,
            tighten: bool = True,
            solver_tag: str = "GUROBI_CMD",
            **solver_kwargs: Any,
    ):
        if not env.loaded:
            raise ValueError(
                "Environment must be loaded before initializing the solver."
            )

        self.set_solver(solver_tag, **solver_kwargs)

        if tighten:
            env.tasks.tighten_bounds(env.current_time)

        self.model, self.variables = self.build_model(env)

    def set_solver(self, solver_tag: str, **solver_kwargs: Any) -> None:
        self.solver = pl.getSolver( # type: ignore[no-untyped-call]
            solver_tag,
            **solver_kwargs,
        )

    @staticmethod
    def build_model(env: SchedulingEnv) -> tuple[LpProblem, PulpVariables]:
        model = LpProblem(env.get_entry(), LpMinimize if env.minimize else LpMaximize)
        _, num_parts = env.tasks.n_tasks, env.tasks.n_parts

        assert num_parts == 1, "This version does not support preemption."

        vars = PulpVariables(env.tasks, env.setup.n_machines)

        export_setup_pulp(env.setup)(model, vars)

        for constraint in env.constraints.values():
            export_constraint_pulp(constraint)(model, vars)

        objective_var = export_objective_pulp(env.objective)(model, vars)

        model.setObjective(objective_var)

        return model, vars

    def solve(self) -> list[tuple[str, int, int]]:
        self.model.solve(self.solver)

        if self.model.status != pl.LpStatusOptimal:
            raise RuntimeError(f"Solver failed with status: {self.model.status}")

        start_times = self.variables.get_start_times()

        return [
            ("execute", task_id, start_time)
            for task_id, start_time in enumerate(start_times)
        ]
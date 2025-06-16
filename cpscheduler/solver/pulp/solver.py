from typing import Any
from collections.abc import Iterable

from pulp import LpProblem, LpSolver, LpMinimize, LpMaximize, LpSolution
import pulp as pl

from cpscheduler.environment.env import SchedulingEnv, ActionType

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from .setup import export_setup_pulp
from .constraint import export_constraint_pulp
from .objective import export_objective_pulp
from .symmetry_breaking import employ_symmetry_breaking_pulp


class PulpSolver:
    solver: LpSolver

    def __init__(
            self,
            env: SchedulingEnv,
            tighten: bool = True,
            solver_tag: str = "GUROBI_CMD",
            timetable: bool = False,
            allow_symmetry_breaking: bool = True,
            **solver_kwargs: Any,
    ):
        if not env.loaded:
            raise ValueError(
                "Environment must be loaded before initializing the solver."
            )

        self.set_solver(solver_tag, **solver_kwargs)

        if tighten:
            env.tasks.tighten_bounds(env.current_time)

        self.model, self.variables = self.build_model(env, timetable, allow_symmetry_breaking)

    def set_solver(self, solver_tag: str, **solver_kwargs: Any) -> None:
        self.solver = pl.getSolver(
            solver_tag,
            **solver_kwargs,
        )

    @staticmethod
    def build_model(
        env: SchedulingEnv,
        timetable: bool = False,
        symmetry_breaking: bool = True
    ) -> tuple[LpProblem, PulpVariables]:
        model = LpProblem(env.get_entry(), LpMinimize if env.minimize else LpMaximize)
        _, num_parts = env.tasks.n_tasks, env.tasks.n_parts

        assert num_parts == 1, "This version does not support preemption."

        variables: PulpVariables = (
            PulpTimetable(env.tasks) if timetable else
            PulpSchedulingVariables(env.tasks)
        )

        export_setup_pulp(env.setup, variables)(model)

        for constraint in env.constraints.values():
            export_constraint_pulp(constraint, variables)(model)

        objective_var = export_objective_pulp(env.objective, variables)(model)
        variables.set_objective(objective_var)

        if symmetry_breaking:
            employ_symmetry_breaking_pulp(env, variables)(model)

        model.setObjective(objective_var)

        return model, variables

    def solve(self) -> tuple[ActionType, float, int]:
        try:
            self.model.solve(self.solver)
        
        except Exception as e:
            if self.model.status <= 0:
                raise RuntimeError(f"Solver failed with status: {LpSolution[self.model.status]}") from e

        if self.model.status <= 0:
            raise RuntimeError(f"Solver failed with status: {LpSolution[self.model.status]}")

        actions = [
            ("execute", task_id, machine_id, start_time) if machine_id != -1 else ("execute", task_id, start_time)
            for task_id, (machine_id, start_time) in enumerate(self.variables.get_assignments())
        ]

        objective_value = self.variables.get_objective_value()

        return actions, objective_value, self.model.status
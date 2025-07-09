from typing import Any, Literal

from pulp import LpProblem, LpSolver, LpMinimize, LpMaximize, LpSolution
import pulp as pl

from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.instructions import ActionType

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from .setup import export_setup_pulp
from .constraint import export_constraint_pulp
from .objective import export_objective_pulp
from .symmetry_breaking import employ_symmetry_breaking_pulp

Formulations = Literal["scheduling", "timetable"]

MAX_ENV_DEPTH = 10  # Maximum depth for the environment wrapping


class PulpSolver:
    solver: LpSolver

    def __init__(
        self,
        env: Any | SchedulingEnv,
        solver_tag: str = "GUROBI_CMD",
        tighten: bool = True,
        formulation: Formulations = "scheduling",
        symmetry_breaking: bool = True,
        integral: bool = False,
        **solver_kwargs: Any,
    ):
        """
        Initialize the solver using PuLP.

        Attributes:
            env: SchedulingEnv
                The environment containing the scheduling problem.
                It must be loaded with an instance before initializing the solver.

            solver_tag: str
                The tag for the PuLP solver to be used (e.g., "GUROBI_CMD", "CPLEX_CMD").

            tighten: bool
                Whether to tighten the bounds of the tasks based on the current time.
                This changes the environment's tasks assuming semi-active scheduling.

        """
        depth = 0
        while not isinstance(env, SchedulingEnv) and depth < MAX_ENV_DEPTH:
            if not hasattr(env, "unwrapped"):
                raise TypeError(
                    f"Expected env to be of type SchedulingEnv or a Wrapped env, got {type(env)} instead."
                )

            if hasattr(env, "core"):
                env = env.core
                break

            env = env.unwrapped
            depth += 1

        if not isinstance(env, SchedulingEnv):
            raise TypeError(
                f"Expected env to be of type SchedulingEnv, got {type(env)} instead."
            )

        self.env = env

        if not self.env.loaded:
            raise ValueError(
                "Environment must be loaded before initializing the solver."
            )

        if self.env.n_parts > 1:
            raise ValueError("This version of the solver does not support preemption.")

        self.set_solver(solver_tag, **solver_kwargs)

        if tighten:
            env.tasks.tighten_bounds(env.current_time)

        self.model, self.variables = self.build_model(
            env, formulation, symmetry_breaking, integral
        )

    def set_solver(self, solver_tag: str, **solver_kwargs: Any) -> None:
        self.solver = pl.getSolver(
            solver_tag,
            **solver_kwargs,
        )

    @staticmethod
    def build_model(
        env: SchedulingEnv,
        formulation: Formulations,
        symmetry_breaking: bool = True,
        integral: bool = False,
    ) -> tuple[LpProblem, PulpVariables]:
        model = LpProblem(
            env.get_entry(), LpMinimize if env.objective.minimize else LpMaximize
        )

        tasks = env.tasks
        data = env.data

        variables: PulpVariables
        if formulation == "timetable":
            variables = PulpTimetable(model, tasks, data, integral)

        elif formulation == "scheduling":
            variables = PulpSchedulingVariables(model, tasks, data, integral)

        if symmetry_breaking:
            employ_symmetry_breaking_pulp(env, model, variables)

        export_setup_pulp(env.setup, variables)(model, tasks, data)

        for constraint in env.constraints.values():
            export_constraint_pulp(constraint, variables)(model, tasks, data)

        objective_var = export_objective_pulp(env.objective, variables)(
            model, tasks, data
        )
        variables.set_objective(objective_var)

        model.setObjective(objective_var)

        return model, variables

    def solve(self) -> tuple[ActionType, float, int]:
        try:
            self.model.solve(self.solver)

        except Exception as e:
            if self.model.status <= 0:
                raise RuntimeError(
                    f"Solver failed with status: {LpSolution[self.model.status]}"
                ) from e

        if self.model.status <= 0:
            raise RuntimeError(
                f"Solver failed with status: {LpSolution[self.model.status]}"
            )

        actions = [
            (
                ("execute", task_id, machine_id, start_time)
                if machine_id != -1
                else ("execute", task_id, start_time)
            )
            for task_id, (machine_id, start_time) in enumerate(
                self.variables.get_assignments()
            )
        ]

        objective_value = self.variables.get_objective_value()

        return actions, objective_value, self.model.status

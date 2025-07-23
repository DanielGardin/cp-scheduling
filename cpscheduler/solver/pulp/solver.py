from typing import Any, Literal
from typing_extensions import Unpack

from pulp import LpProblem, LpSolver, LpMinimize, LpMaximize, LpSolution
import pulp as pl

from cpscheduler.environment import SchedulingEnv, Objective
from cpscheduler.environment.instructions import ActionType
from cpscheduler.common import unwrap_env

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from .setup import export_setup_pulp
from .constraint import export_constraint_pulp
from .objective import export_objective_pulp
from .symmetry_breaking import employ_symmetry_breaking_pulp
from .pulp_utils import SolverConfig, parse_solver_config

Formulations = Literal["scheduling", "timetable"]

class PulpSolver:
    def __init__(
        self,
        env: Any | SchedulingEnv,
        tighten: bool = True,
        formulation: Formulations = "scheduling",
        symmetry_breaking: bool = True,
        integral: bool = False,
        integral_var: bool = True,
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
        self.env = unwrap_env(env)

        if not self.env.loaded:
            raise ValueError(
                "Environment must be loaded before initializing the solver."
            )

        if self.env.n_parts > 1:
            raise ValueError("This version of the solver does not support preemption.")

        if tighten:
            env.tasks.tighten_bounds(env.current_time)

        self._solver: LpSolver | None = None
        self.model, self.variables = self.build_model(
            env, formulation, symmetry_breaking, integral, integral_var
        )

    @classmethod
    def available_solvers(cls) -> list[str]:
        solvers_list: list[str] = pl.listSolvers(onlyAvailable=True)

        return solvers_list

    def set_solver(
        self,
        solver_tag: str,
        **solver_kwargs: Unpack[SolverConfig],
    ) -> None:
        config = parse_solver_config(solver_kwargs)

        self._solver = pl.getSolver(
            solver_tag,
            **config,
        )

    @staticmethod
    def build_model(
        env: SchedulingEnv,
        formulation: Formulations,
        symmetry_breaking: bool = True,
        integral: bool = False,
        integral_var: bool = True,
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
            variables = PulpSchedulingVariables(model, tasks, data, integral, integral_var)

        if symmetry_breaking:
            employ_symmetry_breaking_pulp(env, model, variables)

        export_setup_pulp(env.setup, variables)(model, tasks, data)

        for constraint in env.constraints.values():
            export_constraint_pulp(constraint, variables)(model, tasks, data)

        if type(env.objective) is not Objective:
            objective_var = export_objective_pulp(env.objective, variables)(
                model, tasks, data
            )
            variables.set_objective(objective_var)

            model.setObjective(objective_var)

        return model, variables

    def solve(
        self,
        solver_tag: str | None = None,
        **solver_kwargs: Unpack[SolverConfig],
    ) -> tuple[ActionType, float, int]:
        if solver_tag is not None:
            self.set_solver(solver_tag, **solver_kwargs)

        self.model.solve(self._solver)

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

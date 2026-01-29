from typing import Any, Literal
from collections.abc import Sequence
from typing_extensions import Unpack

from copy import deepcopy

from pulp import LpProblem, LpSolver, LpMinimize, LpMaximize, LpSolution
import pulp as pl

from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.objectives import Objective
from cpscheduler.environment.instructions import SingleAction, ActionType
from cpscheduler.common import unwrap_env

from cpscheduler.solver.pulp.tasks import (
    PulpVariables,
    PulpSchedulingVariables,
    PulpTimetable,
)
from cpscheduler.solver.pulp.setup import export_setup_pulp
from cpscheduler.solver.pulp.constraint import export_constraint_pulp
from cpscheduler.solver.pulp.objective import export_objective_pulp
from cpscheduler.solver.pulp.symmetry_breaking import employ_symmetry_breaking_pulp
from cpscheduler.solver.pulp.pulp_utils import SolverConfig, parse_solver_config

Formulations = Literal["scheduling", "timetable"]


class PulpSolver:
    model: LpProblem
    variables: PulpVariables

    _solver: LpSolver | None
    _config: SolverConfig

    symmetry_breaking: bool
    _built: bool

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
                This changes the environment's tasks assuming semi-active scheduling and
                regular objective.

        """
        self.env = unwrap_env(env)

        if not self.env.state.loaded:
            raise ValueError("Environment must be loaded before initializing the solver.")

        self._solver = None
        self._config = {}

        sense = LpMinimize if self.env.objective.minimize else LpMaximize
        self.model = LpProblem(env.get_entry(), sense)

        if formulation == "timetable":
            self.variables = PulpTimetable(self.model, self.env.state, integral)

        elif formulation == "scheduling":
            self.variables = PulpSchedulingVariables(
                self.model, self.env.state, integral, integral_var
            )

        if tighten:
            self.warm_start([("submit", task.task_id) for task in self.env.state.awaiting_tasks])

        self.symmetry_breaking = symmetry_breaking
        self._built = False

    @classmethod
    def available_solvers(cls) -> list[str]:
        solvers_list: list[str] = pl.listSolvers(onlyAvailable=True)

        return solvers_list

    def set_solver(
        self,
        solver_tag: str,
        **solver_kwargs: Unpack[SolverConfig],
    ) -> None:
        config = parse_solver_config(self._config | solver_kwargs)

        self._solver = pl.getSolver(
            solver_tag,
            **config,
        )

    def build(self) -> None:
        env = self.env
        state = self.env.state

        export_setup_pulp(env.setup, self.variables)(self.model, state)

        for constraint in env.setup_constraints:
            export_constraint_pulp(constraint, self.variables)(self.model, state)

        for constraint in env.constraints:
            export_constraint_pulp(constraint, self.variables)(self.model, state)

        if type(env.objective) is not Objective:
            objective_var = export_objective_pulp(env.objective, self.variables)(self.model, state)
            self.variables.set_objective(objective_var)

            self.model.setObjective(objective_var)

        if self.symmetry_breaking:
            employ_symmetry_breaking_pulp(self.env, self.model, self.variables)

        self._built = True

    def set_horizon(self, horizon: int) -> None:
        """
        Set the time horizon for the scheduling problem.

        Args:
            horizon: int
                The time horizon to be set.
        """
        self.variables.set_horizon(horizon)

    def warm_start(self, action: ActionType) -> None:
        """
        Set the initial values for the variables based on the provided action.

        Args:
            action: ActionType
                The action used for warm starting the solver.
        """
        env_copy = deepcopy(self.env)

        if env_copy.force_reset:
            env_copy.reset()

        env_copy.step(action)
        self.variables.warm_start(env_copy)

        self._config["warm_start"] = True

    def solve(
        self,
        solver_tag: str | None = None,
        **solver_kwargs: Unpack[SolverConfig],
    ) -> tuple[Sequence[SingleAction], float, int]:
        if solver_tag is not None:
            self.set_solver(solver_tag, **solver_kwargs)

        if not self._built:
            self.build()

        self.model.solve(self._solver)

        assert isinstance(self.model.status, int)
        status = self.model.status

        if status <= 0:
            raise RuntimeError(f"Solver failed with status: {LpSolution[status]}")

        actions: list[tuple[str, int, int, int]] = []

        for task in self.env.state.awaiting_tasks:
            task_id = task.task_id
            assignment = self.variables.get_assigment(task_id)

            actions.append(("execute", task_id, *assignment))

        actions.sort(key=lambda x: (x[-1], x[1]))

        objective_value = self.variables.get_objective_value()

        return actions, objective_value, status


if __name__ == "__main__":
    print("Available MiniZinc solvers:", PulpSolver.available_solvers())

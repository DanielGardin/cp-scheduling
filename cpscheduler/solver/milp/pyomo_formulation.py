"""Pyomo-based MILP formulation for scheduling problems."""

import logging
from typing import Any, Final, Literal, TypeAlias

import pyomo.environ as pyo
from pyomo.core.base.objective import ObjectiveSense
from pyomo.core.base.var import VarData
from pyomo.core.expr.relational_expr import RelationalExpression
from pyomo.opt.base.solvers import OptSolver
from pyomo.opt.results.results_ import SolverResults
from typing_extensions import assert_never, override

from cpscheduler.environment import SchedulingEnv
from cpscheduler.solver.formulation import Formulation

ScalarValue: TypeAlias = int | float
PYOMO_PARAM: TypeAlias = ScalarValue | VarData
PYOMO_CONSTRAINT: TypeAlias = bool | RelationalExpression

# Maps solver tag to the option name it uses for time limits.
_SOLVER_TIME_LIMIT_OPTION: dict[str, str] = {
    "cplex": "timelimit",
    "cplex_direct": "timelimit",
    "cplex_persistent": "timelimit",
    "gurobi": "TimeLimit",
    "gurobi_direct": "TimeLimit",
    "gurobi_persistent": "TimeLimit",
    "xpress": "maxtime",
    "xpress_direct": "maxtime",
    "mosek": "MSK_DPAR_OPTIMIZER_MAX_TIME",
    "mosek_direct": "MSK_DPAR_OPTIMIZER_MAX_TIME",
    "highs": "time_limit",
    "scip": "limits/time",
    "cbc": "sec",
    "glpk": "tmlim",
}

# Solver preference order for automatic selection.
# Direct/persistent interfaces are preferred over executable-based ones
# as they avoid file I/O and subprocess overhead.
_SOLVER_PREFERENCE: list[str] = [
    # Commercial: CPLEX
    "cplex_direct",
    "cplex_persistent",
    "cplex",
    # Commercial: Gurobi
    "gurobi_direct",
    "gurobi_persistent",
    "gurobi",
    # Commercial: Xpress
    "xpress_direct",
    "xpress",
    # Commercial: MOSEK
    "mosek_direct",
    "mosek",
    # Open-source: HiGHS (best open-source MILP as of 2024)
    "highs",
    # Open-source: SCIP
    "scip",
    # Open-source: CBC
    "cbc",
    # Open-source: GLPK (weakest, last resort)
    "glpk",
]


def find_available_solver() -> str | None:
    """Find the first available solver from the preference list.

    The order of solvers is defined as follows:
    1. CPLEX (direct/persistent interfaces preferred)
    2. Gurobi (direct/persistent interfaces preferred)
    3. Xpress (direct interface preferred)
    4. MOSEK (direct interface preferred)
    5. HiGHS
    6. SCIP
    7. CBC
    8. GLPK

    Returns
    -------
    tag: str | None
        The tag of the first available solver, or None if no solver is found.

    """
    for tag in _SOLVER_PREFERENCE:
        try:
            if pyo.SolverFactory(tag).available():
                return tag

        except Exception:
            continue

    return None


DEFAULT_BIG_M: Final[float] = 1e6


Strategies = Literal["big_m", "convex_hull", "gdp_big_m"]


class PyomoFormulation(Formulation[SolverResults]):
    """Generic pyomo-based MILP model formulation.

    This class is a base formulation that provides common functionality for
    model building, variable management, constraint addition, and solving using
    Pyomo.

    Specific formulations can inherit from this class and implement the
    `build` method to construct the model according to the scheduling problem's
    requirements.
    """

    model: pyo.ConcreteModel

    _sense: ObjectiveSense
    _solver: OptSolver
    _result: SolverResults
    _objective_expr: PYOMO_PARAM

    _variables: dict[str, VarData]
    _disjunctive_strategy: Strategies

    def __init__(self, disjunctive_strategy: Strategies = "big_m"):
        self._disjunctive_strategy = disjunctive_strategy

    @override
    def initialize_model(self, env: SchedulingEnv) -> None:
        self._objective_expr = 0

        self.model = pyo.ConcreteModel(name=env.get_entry())
        self._sense = pyo.minimize if env.objective.minimize else pyo.maximize
        self.model.objective = pyo.Objective(expr=0, sense=self._sense)

        self._variables = {}

    @override
    def post_build(self) -> None:
        return super().post_build()

    def solve(
        self,
        solver_tag: str | None = None,
        quiet: bool = False,
        time_limit: float | None = None,
        keep_files: bool = False,
        **solver_kwargs: Any,
    ) -> SolverResults:
        """Solve the MILP model using the specified solver and options.

        Parameters
        ----------
        solver_tag: str | None, optional
            The tag of the solver to use.
            If None, the first available solver will be automatically selected.

        quiet: bool, optional
            If True, suppress solver output. Default is False.

        time_limit: float | None, optional
            Time limit for the solver in seconds. If None, no time limit is applied.

        keep_files: bool, optional
            If True, keep temporary files created by the solver. Default is False.

        **solver_kwargs: Any
            Additional keyword arguments to pass to the Pyomo SolverFactory.

        """
        if solver_tag is None:
            solver_tag = find_available_solver()
            if solver_tag is None:
                raise RuntimeError("No Pyomo-compatible MILP solver found.")

            logging.info(f"No solver specified, using {solver_tag}.")

        self._solver = pyo.SolverFactory(solver_tag, **solver_kwargs)

        if time_limit is not None:
            option_name = _SOLVER_TIME_LIMIT_OPTION.get(solver_tag)

            if option_name is not None:
                self._solver.options[option_name] = time_limit

            else:
                logging.warning(
                    f"Time limit not applied: no known option name for solver '{solver_tag}'. "
                    f"Add it to _SOLVER_TIME_LIMIT_OPTION."
                )

        self._result = self._solver.solve(
            self.model,
            tee=not quiet,
            keepfiles=keep_files,
        )

        return self._result

    def set_objective(self, expr: PYOMO_PARAM) -> None:
        """Set the objective expression of the model."""
        self._objective_expr = expr
        self.model.del_component(self.model.objective)
        self.model.objective = pyo.Objective(expr=expr, sense=self._sense)

    def add_var(
        self,
        name: str,
        lb: float | int | None = None,
        ub: float | int | None = None,
        binary: bool = False,
    ) -> VarData:
        """Add a variable to the model with the given name and properties.

        Parameters
        ----------
        name: str
            The name of the variable. Must be unique within the model.

        lb: float | int | None, optional
            The lower bound of the variable. If None, no lower bound is applied.

        ub: float | int | None, optional
            The upper bound of the variable. If None, no upper bound is applied.

        binary: bool, optional
            Whether the variable is binary (0 or 1). If True, lb and ub are
            ignored and the variable is constrained to be binary.

        """
        if name in self._variables:
            raise ValueError(f"Variable '{name}' already exists.")

        if binary:
            var = pyo.Var(within=pyo.Binary, bounds=(0, 1))
        else:
            var = pyo.Var(within=pyo.Reals, bounds=(lb, ub))

        self.model.add_component(name, var)

        assert isinstance(var, VarData), (
            f"Unreachable code: {var} is not indexable."
        )
        self._variables[name] = var

        return var

    def add_constraint(self, constraint: PYOMO_CONSTRAINT, name: str) -> None:
        """Add a constraint to the model.

        Parameters
        ----------
        constraint: PYOMO_CONSTRAINT
            The constraint to add. Can be a boolean (trivial constraint) or a
            Pyomo relational expression.

        name: str
            The name of the constraint. Must be unique within the model.

        Raises
        ------
        ValueError
            If the constraint is trivially False, or if the name is not unique.

        """
        if isinstance(constraint, bool):
            if not constraint:
                raise ValueError(
                    f"Constraint '{name}' is trivially False; adding it invalidates the model."
                )
            return

        self.model.add_component(name, pyo.Constraint(expr=constraint))

    def max_expr(
        self,
        decision_vars: list[PYOMO_PARAM],
        name: str,
        exact: bool = False,
    ) -> VarData:
        """Create an auxiliary variable representing the maximum of a list of decision variables.

        This method creates a new variable `z` and adds constraints `z >= v` for
        each variable `v` in `decision_vars`.

        Parameters
        ----------
        decision_vars: list[PYOMO_PARAM]
            The list of decision variables to take the maximum over.

        name: str
            The name of the auxiliary variable to create.

        exact: bool, default=False
            Whether the maximum variable should be exact.
            When False, the constraints only ensure that `z` is an upper bound
            on the decision variables, which can be a relaxation.
            If the caller minimizes `z` in the objective, it will be driven down
            to the maximum of the decision variables, making it exact even when
            `exact=False`.

        Returns
        -------
        VarData
            The auxiliary variable representing the (upper bound for the)
            maximum of the decision variables.

        """
        if len(decision_vars) == 0:
            raise ValueError(
                "Cannot take the maximum of an empty list of variables."
            )

        lb = max(self.get_lb(var) for var in decision_vars)
        ub = max(self.get_ub(var) for var in decision_vars)

        max_var = self.add_var(name, lb=lb, ub=ub, binary=False)

        for i, var in enumerate(decision_vars):
            self.add_constraint(max_var >= var, f"{name}_lb_{i}")

        if not exact:
            return max_var

        binaries = [
            self.add_var(f"{name}_is_max_{i}", binary=True)
            for i in range(len(decision_vars))
        ]

        self.add_constraint(
            pyo.quicksum(binaries) == 1, f"{name}_exact_one_max"
        )

        for i, (var, binary) in enumerate(
            zip(decision_vars, binaries, strict=True)
        ):
            self.implication(
                (binary,), (max_var, "==", var), f"{name}_exact_max_{i}"
            )

        return max_var

    def implication(
        self,
        antecedent: tuple[PYOMO_PARAM, ...],
        consequent: tuple[PYOMO_PARAM, Literal["==", "<=", ">="], PYOMO_PARAM],
        name: str | None = None,
    ) -> None:
        """Add a linear constraint representing an implication.

        This method adds constraints to the model so that, when all variables in
        the `antecedent` are equal to 1 (True), the `consequent` constraint is
        enforced.

        Parameters
        ----------
        antecedent: tuple[PYOMO_PARAM, ...]
            A tuple of variables representing the antecedent conditions.

        consequent: tuple[PYOMO_PARAM, Literal["==", "<=", ">="], PYOMO_PARAM]
            A tuple representing the consequent constraint in the form
            (lhs, operator, rhs), where `operator` is one of "==", "<=", ">=".

        name: str | None, optional
            An optional name for the constraints added by this implication. If
            None, default names will be generated based on the type of implication.

        """
        premises: list[PYOMO_PARAM] = []

        for var in antecedent:
            truth = self._is_true(var == 1)
            if truth is True:
                continue

            if truth is False:
                return

            premises.append(var)

        lhs, operator, rhs = consequent

        if not premises:
            self._add_direct_consequent(lhs, operator, rhs, name)
            return

        premise_sum: PYOMO_PARAM = pyo.quicksum(premises)
        n_vars = len(premises)

        gap_ub = self.get_ub(lhs) - self.get_lb(rhs)
        gap_lb = self.get_lb(lhs) - self.get_ub(rhs)

        if operator == "==":
            self.add_constraint(
                lhs - rhs <= (n_vars - premise_sum) * gap_ub,
                f"{name}_le" if name else "implication_eq_le",
            )
            self.add_constraint(
                lhs - rhs >= (n_vars - premise_sum) * gap_lb,
                f"{name}_ge" if name else "implication_eq_ge",
            )

        elif operator == "<=":
            self.add_constraint(
                lhs - rhs <= (n_vars - premise_sum) * gap_ub,
                name or "implication_le",
            )

        elif operator == ">=":
            self.add_constraint(
                lhs - rhs >= (n_vars - premise_sum) * gap_lb,
                name or "implication_ge",
            )

        else:
            assert_never(operator)

    def _add_direct_consequent(
        self,
        lhs: PYOMO_PARAM,
        operator: Literal["==", "<=", ">="],
        rhs: PYOMO_PARAM,
        name: str | None,
    ) -> None:
        if operator == "==":
            self.add_constraint(lhs == rhs, name or "direct_eq")

        elif operator == "<=":
            self.add_constraint(lhs <= rhs, name or "direct_le")

        elif operator == ">=":
            self.add_constraint(lhs >= rhs, name or "direct_ge")

    def get_value(self, param: PYOMO_PARAM) -> float:
        """Get the numerical value of a Pyomo parameter or variable."""
        if isinstance(param, int | float):
            return float(param)

        value = param.value
        if value is None:
            raise RuntimeError(
                f"Unreachable code: Pyomo should have handled an exception for "
                f"parameter {param}."
            )

        return float(value)

    def set_initial_value(self, param: PYOMO_PARAM, value: float | int) -> None:
        """Set the initial value of a Pyomo variable or parameter."""
        if isinstance(param, int | float):
            if param != value:
                raise ValueError(
                    f"Cannot set initial value of constant {param} to {value}."
                )
            return

        param.set_value(value)

    def get_ub(self, param: PYOMO_PARAM) -> float:
        """Get the upper bound of a Pyomo parameter or variable."""
        if isinstance(param, int | float):
            return float(param)

        ub = param.ub

        assert ub is not None, f"Cannot obtain an upper bound for {param}."

        return float(ub)

    def get_lb(self, param: PYOMO_PARAM) -> float:
        """Get the lower bound of a Pyomo parameter or variable."""
        if isinstance(param, int | float):
            return float(param)

        lb = param.lb

        assert lb is not None, f"Cannot obtain a lower bound for {param}."

        if hasattr(param, "lb") and param.lb is not None:
            return float(param.lb)

        return float(lb)

    def set_ub(self, param: PYOMO_PARAM, ub: float | int) -> None:
        """Set the upper bound of a Pyomo variable or parameter."""
        if isinstance(param, int | float):
            if param > ub:
                raise ValueError(
                    f"Cannot set upper bound of constant {param} to {ub}."
                )
            return

        param.setub(ub)

    def _is_true(
        self, condition: PYOMO_CONSTRAINT | PYOMO_PARAM
    ) -> bool | None:
        if isinstance(condition, bool):
            return condition

        if isinstance(condition, int | float):
            return bool(condition)

        if isinstance(condition, RelationalExpression):
            if condition.is_constant():
                return bool(pyo.value(condition))

        else:
            value = condition.value

            if value is not None:
                return bool(value)

        return None

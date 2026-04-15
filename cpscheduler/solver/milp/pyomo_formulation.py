import logging
from typing import Final, TypeAlias, Literal
from typing_extensions import assert_never

import pyomo.environ as pyo                                       # type: ignore[import-untyped]
from pyomo.core.base.objective import ObjectiveSense              # type: ignore[import-untyped]
from pyomo.core.expr.relational_expr import RelationalExpression  # type: ignore[import-untyped]
from pyomo.opt.base.solvers import OptSolver                      # type: ignore[import-untyped]
from pyomo.opt.results.results_ import SolverResults              # type: ignore[import-untyped]
from pyomo.core.base.var import NumericValue, VarData             # type: ignore[import-untyped]

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
    for tag in _SOLVER_PREFERENCE:
        try:
            if pyo.SolverFactory(tag).available():
                return tag

        except Exception:
            continue

    return None

DEFAULT_BIG_M: Final[float] = 1e6

class PyomoFormulation(Formulation):
    model: pyo.ConcreteModel

    _sense: ObjectiveSense
    _solver: OptSolver
    _result: SolverResults
    _objective_expr: PYOMO_PARAM

    _variables: dict[str, VarData]

    def initialize_pyomo_model(self, name: str, minimize: bool) -> None:
        self._objective_expr = 0

        self.model = pyo.ConcreteModel(name=name)
        self._sense = pyo.minimize if minimize else pyo.maximize
        self.model.objective = pyo.Objective(expr=0, sense=self._sense)

        self._variables = {}

    def solve(
        self,
        solver_tag: str | None = None,
        quiet: bool = False,
        time_limit: float | None = None,
        keep_files: bool = False,
        **solver_kwargs: object,
    ) -> str:
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

        termination = self._result.solver.termination_condition
        return str(termination)

    def set_objective(self, expr: PYOMO_PARAM) -> None:
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
        if name in self._variables:
            raise ValueError(f"Variable '{name}' already exists.")

        if binary:
            var = pyo.Var(within=pyo.Binary, bounds=(0, 1))
        else:
            var = pyo.Var(within=pyo.Reals, bounds=(lb, ub))

        self.model.add_component(name, var)

        assert isinstance(var, VarData), f"Unreachable code: {var} is not indexable."
        self._variables[name] = var

        return var

    def add_constraint(self, constraint: PYOMO_CONSTRAINT, name: str) -> None:
        if isinstance(constraint, bool):
            if not constraint:
                raise ValueError(
                    f"Constraint '{name}' is trivially False; adding it invalidates the model."
                )
            return

        self.model.add_component(
            name, pyo.Constraint(expr=constraint)
        )

    def max_expr(
        self,
        decision_vars: list[PYOMO_PARAM],
        name: str,
        minimized: bool = True,
    ) -> VarData:
        """
        Returns an auxiliary variable `z` constrained by `z >= v` for each v
        in `decision_vars`.

        When `minimized=True` (default), the objective drives `z` down to
        `max(decision_vars)`, making this exact. When `minimized=False`, `z`
        is only a valid lower envelope and callers must enforce tightness
        through additional constraints or accept the relaxation.
        """
        max_var = self.add_var(name, lb=None, ub=None, binary=False)

        for i, var in enumerate(decision_vars):
            self.add_constraint(max_var >= var, f"{name}_lb_{i}")

        return max_var

    def implication(
        self,
        antecedent: tuple[PYOMO_PARAM, ...],
        consequent: tuple[PYOMO_PARAM, Literal["==", "<=", ">="], PYOMO_PARAM],
        name: str | None = None,
    ) -> None:
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
        if isinstance(param, (int, float)):
            return float(param)

        value = param.value
        if value is None:
            raise RuntimeError(
                f"Unreachable code: Pyomo should have handled an exception for "
                f"parameter {param}."
            )

        return float(value)

    def set_initial_value(self, param: PYOMO_PARAM, value: float | int) -> None:
        if isinstance(param, (int, float)):
            if param != value:
                raise ValueError(
                    f"Cannot set initial value of constant {param} to {value}."
                )
            return

        param.set_value(value)

    def get_ub(self, param: PYOMO_PARAM) -> float:
        if isinstance(param, (int, float)):
            return float(param)

        ub = param.ub

        assert ub is not None, f"Cannot obtain an upper bound for {param}."

        return float(ub)

    def get_lb(self, param: PYOMO_PARAM) -> float:
        if isinstance(param, (int, float)):
            return float(param)

        lb = param.lb

        assert lb is not None, f"Cannot obtain a lower bound for {param}."


        if hasattr(param, "lb") and param.lb is not None:
            return float(param.lb)

        return float(lb)

    def set_ub(self, param: PYOMO_PARAM, ub: float | int) -> None:
        if isinstance(param, (int, float)):
            if param > ub:
                raise ValueError(
                    f"Cannot set upper bound of constant {param} to {ub}."
                )
            return

        param.setub(ub)

    def _is_true(self, condition: PYOMO_CONSTRAINT | PYOMO_PARAM) -> bool | None:
        if isinstance(condition, bool):
            return condition

        if isinstance(condition, (int, float)):
            return bool(condition)

        if isinstance(condition, RelationalExpression):
            if condition.is_constant():
                return bool(pyo.value(condition))

        else:
            value = condition.value

            if value is not None:
                return bool(value)

        return None

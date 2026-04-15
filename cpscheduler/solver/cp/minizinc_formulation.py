import datetime
import logging
from dataclasses import dataclass
from typing import Any, TypeAlias
from collections.abc import Sequence

import minizinc  # type: ignore[import-untyped]

from cpscheduler.solver.formulation import Formulation

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ScalarValue: TypeAlias = int | float | bool

# A MiniZinc parameter is either a concrete scalar or a MiniZinc expression
# string (variable name, array access like "x[i]", or arithmetic expression).
MZN_PARAM: TypeAlias = ScalarValue | str
MZN_CONSTRAINT: TypeAlias = bool | str


# ---------------------------------------------------------------------------
# Variable descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntVarArray:
    """
    A MiniZinc array of integer variables: array[1..size] of var lb..ub: name.
    Element access: var_array[i] returns the MiniZinc expression "name[i]".
    """
    name: str
    size: int
    lb: int
    ub: int

    def __getitem__(self, i: int) -> str:
        """1-indexed, matching MiniZinc convention."""
        return f"{self.name}[{i}]"

    def as_mzn(self) -> str:
        return self.name


@dataclass(frozen=True)
class BoolVarArray:
    """
    A MiniZinc array of boolean variables: array[1..size] of var bool: name.
    """
    name: str
    size: int

    def __getitem__(self, i: int) -> str:
        return f"{self.name}[{i}]"

    def as_mzn(self) -> str:
        return self.name


@dataclass(frozen=True)
class IntVar:
    """A scalar MiniZinc integer variable."""
    name: str
    lb: int
    ub: int

    def as_mzn(self) -> str:
        return self.name


@dataclass(frozen=True)
class BoolVar:
    """A scalar MiniZinc boolean variable."""
    name: str

    def as_mzn(self) -> str:
        return self.name


@dataclass(frozen=True)
class IntervalVarArray:
    """
    A family of interval variables represented as four linked MiniZinc arrays:
    start, end, duration, and present — all indexed 1..size.

    MiniZinc has no native interval variable type outside of specialized
    libraries with limited solver support. This models intervals explicitly
    with four arrays and linking constraints, enabling use of cumulative and
    disjunctive globals universally.
    """
    start: IntVarArray
    end: IntVarArray
    duration: IntVarArray
    present: BoolVarArray
    optional: bool

    @property
    def size(self) -> int:
        return self.start.size

    def __getitem__(self, i: int) -> tuple[str, str, str, str]:
        """Returns (start[i], end[i], duration[i], present[i]) — 1-indexed."""
        return (
            self.start[i],
            self.end[i],
            self.duration[i],
            self.present[i],
        )


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

_SOLVER_PREFERENCE: list[str] = [
    "com.google.ortools.sat",   # OR-Tools CP-SAT via MiniZinc
    "org.chuffed.chuffed",      # Chuffed — strong for scheduling with LNS
    "org.gecode.gecode",        # Gecode — default, most stable
    "org.minizinc.mip.highs",   # HiGHS via MiniZinc MIP backend
    "org.minizinc.mip.cplex",   # CPLEX via MiniZinc MIP backend
    "org.minizinc.mip.gurobi",  # Gurobi via MiniZinc MIP backend
]


def find_available_solver() -> str | None:
    for tag in _SOLVER_PREFERENCE:
        try:
            minizinc.Solver.lookup(tag)
            return tag
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Base formulation
# ---------------------------------------------------------------------------

class MiniZincFormulation(Formulation):
    """
    Base class for MiniZinc-based formulations.

    Builds a MiniZinc model as a text string using array-first variable
    declarations. Arrays are the primary interface because MiniZinc global
    constraints (cumulative, disjunctive, alldifferent, lex_less, circuit)
    all operate on arrays, and indexed array families are more natural for
    scheduling than flat scalar collections.

    Variable references are MiniZinc expression strings. Constraints are
    MiniZinc expression strings emitted into the model. This mirrors the
    MZN_PARAM / add_* / add_constraint interface of PyomoFormulation while
    respecting MiniZinc's array-centric semantics.
    """

    _model_str: str
    _objective_expr: str
    _minimize: bool
    _var_id: int
    _result: minizinc.Result | None

    def __init__(self) -> None:
        self._model_str = ""
        self._objective_expr = "0"
        self._minimize = True
        self._var_id = 0
        self._result = None

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def initialize_minizinc_model(self, minimize: bool) -> None:
        self._var_id = 0
        self._model_str = ""
        self._objective_expr = "0"
        self._minimize = minimize
        self._result = None

    def _solve_annotation(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._var_id += 1
        return self._var_id

    def _unique(self, name: str) -> str:
        return f"{name}_{self._next_id()}"

    def _emit(self, line: str) -> None:
        self._model_str += line + "\n"

    def _to_mzn(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        if hasattr(value, "as_mzn"):
            return value.as_mzn()  # type: ignore[no-any-return]

        return str(value)

    # ------------------------------------------------------------------
    # Array variable creation — primary interface
    # ------------------------------------------------------------------

    def add_int_var_array(
        self,
        name: str,
        size: int,
        lb: int,
        ub: int,
    ) -> IntVarArray:
        """
        Declare: array[1..size] of var lb..ub: name;
        Returns an IntVarArray whose elements are accessible as name[i].
        """
        var_name = self._unique(name)
        self._emit(f"array[1..{size}] of var {lb}..{ub}: {var_name};")
        return IntVarArray(name=var_name, size=size, lb=lb, ub=ub)

    def add_bool_var_array(
        self,
        name: str,
        size: int,
    ) -> BoolVarArray:
        """
        Declare: array[1..size] of var bool: name;
        """
        var_name = self._unique(name)
        self._emit(f"array[1..{size}] of var bool: {var_name};")
        return BoolVarArray(name=var_name, size=size)

    def add_int_var_2d(
        self,
        name: str,
        rows: int,
        cols: int,
        lb: int,
        ub: int,
    ) -> str:
        """
        Declare: array[1..rows, 1..cols] of var lb..ub: name;
        Returns the array name. Element access: "name[i,j]".
        """
        var_name = self._unique(name)
        self._emit(
            f"array[1..{rows}, 1..{cols}] of var {lb}..{ub}: {var_name};"
        )
        return var_name

    def add_bool_var_2d(
        self,
        name: str,
        rows: int,
        cols: int,
    ) -> str:
        """
        Declare: array[1..rows, 1..cols] of var bool: name;
        Returns the array name. Element access: "name[i,j]".
        """
        var_name = self._unique(name)
        self._emit(
            f"array[1..{rows}, 1..{cols}] of var bool: {var_name};"
        )
        return var_name

    def add_interval_var_array(
        self,
        name: str,
        size: int,
        start_lb: int,
        start_ub: int,
        end_lb: int,
        end_ub: int,
        duration_lb: int,
        duration_ub: int,
        optional: bool = False,
    ) -> IntervalVarArray:
        """
        Declare four linked arrays representing a family of interval variables:

            array[1..size] of var start_lb..start_ub: name_start;
            array[1..size] of var end_lb..end_ub:     name_end;
            array[1..size] of var dur_lb..dur_ub:     name_dur;
            array[1..size] of var bool:               name_present;

        When optional=False, presence is fixed to true and the linking
        constraint end[i] = start[i] + dur[i] is unconditional.
        When optional=True, the linking constraint is guarded by presence[i].
        """
        uid = self._next_id()
        start_name = f"{name}_start_{uid}"
        end_name = f"{name}_end_{uid}"
        dur_name = f"{name}_dur_{uid}"
        present_name = f"{name}_present_{uid}"

        self._emit(
            f"array[1..{size}] of var {start_lb}..{start_ub}: {start_name};"
        )
        self._emit(
            f"array[1..{size}] of var {end_lb}..{end_ub}: {end_name};"
        )
        self._emit(
            f"array[1..{size}] of var {duration_lb}..{duration_ub}: {dur_name};"
        )

        if optional:
            self._emit(f"array[1..{size}] of var bool: {present_name};")
            self._emit(
                f"constraint forall(i in 1..{size})("
                f"{present_name}[i] -> "
                f"({end_name}[i] = {start_name}[i] + {dur_name}[i])"
                f");"
            )
        else:
            self._emit(
                f"array[1..{size}] of bool: {present_name} = "
                f"[true | i in 1..{size}];"
            )
            self._emit(
                f"constraint forall(i in 1..{size})("
                f"{end_name}[i] = {start_name}[i] + {dur_name}[i]"
                f");"
            )

        start_arr = IntVarArray(
            name=start_name, size=size, lb=start_lb, ub=start_ub
        )
        end_arr = IntVarArray(
            name=end_name, size=size, lb=end_lb, ub=end_ub
        )
        dur_arr = IntVarArray(
            name=dur_name, size=size, lb=duration_lb, ub=duration_ub
        )
        present_arr = BoolVarArray(name=present_name, size=size)

        return IntervalVarArray(
            start=start_arr,
            end=end_arr,
            duration=dur_arr,
            present=present_arr,
            optional=optional,
        )

    # ------------------------------------------------------------------
    # Scalar variable creation — for genuinely singleton variables
    # ------------------------------------------------------------------

    def add_int_var(self, name: str, lb: int, ub: int) -> IntVar:
        var_name = self._unique(name)
        self._emit(f"var {lb}..{ub}: {var_name};")
        return IntVar(name=var_name, lb=lb, ub=ub)

    def add_bool_var(self, name: str) -> BoolVar:
        var_name = self._unique(name)
        self._emit(f"var bool: {var_name};")
        return BoolVar(name=var_name)

    # ------------------------------------------------------------------
    # Parameter arrays (fixed data, not decision variables)
    # ------------------------------------------------------------------

    def add_int_param_array(
        self,
        name: str,
        values: list[int],
    ) -> str:
        """
        Declare a fixed integer parameter array from a Python list.
        Returns the MiniZinc array name.
        """
        var_name = self._unique(name)
        mzn_values = "[" + ", ".join(str(v) for v in values) + "]"
        self._emit(
            f"array[1..{len(values)}] of int: {var_name} = {mzn_values};"
        )
        return var_name

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def add_constraint(self, constraint: MZN_CONSTRAINT, name: str) -> None:
        if isinstance(constraint, bool):
            if not constraint:
                raise ValueError(
                    f"Constraint '{name}' is trivially False; "
                    "adding it invalidates the model."
                )
            return
        self._emit(f"constraint {constraint}; % {name}")

    def add_forall_constraint(
        self,
        index: str,
        size: int,
        body: str,
        name: str,
    ) -> None:
        """
        Emit: constraint forall(index in 1..size)(body);
        Use when the constraint body is parameterized by index.
        """
        self._emit(
            f"constraint forall({index} in 1..{size})({body}); % {name}"
        )

    def no_overlap(
        self,
        intervals: IntervalVarArray,
        name: str,
    ) -> None:
        """
        Post a disjunctive no-overlap constraint.

        For non-optional intervals, uses MiniZinc's disjunctive global
        directly. For optional intervals, falls back to cumulative with
        bool2int(present[i]) demands, since disjunctive has no native
        optional support across all solvers.
        """
        if intervals.optional:
            demands = (
                f"[bool2int({intervals.present.name}[i]) | i in 1..{intervals.size}]"
            )
            self._emit(
                f"constraint cumulative("
                f"{intervals.start.name}, "
                f"{intervals.duration.name}, "
                f"{demands}, 1"
                f"); % {name}"
            )
        else:
            self._emit(
                f"constraint disjunctive("
                f"{intervals.start.name}, "
                f"{intervals.duration.name}"
                f"); % {name}"
            )

    def cumulative(
        self,
        intervals: IntervalVarArray,
        demands: list[MZN_PARAM] | str,
        capacity: MZN_PARAM,
        name: str,
    ) -> None:
        """
        Post a cumulative resource constraint.
        demands can be a Python list of MZN_PARAM or a MiniZinc array name string.
        """
        if isinstance(demands, list):
            dem_str = "[" + ", ".join(str(d) for d in demands) + "]"
        else:
            dem_str = demands

        self._emit(
            f"constraint cumulative("
            f"{intervals.start.name}, "
            f"{intervals.duration.name}, "
            f"{dem_str}, "
            f"{capacity}"
            f"); % {name}"
        )

    def alldifferent(self, array: IntVarArray | str, name: str) -> None:
        arr = array if isinstance(array, str) else array.name
        self._emit(f"constraint alldifferent({arr}); % {name}")

    def lex_less(
        self,
        a: IntVarArray | str,
        b: IntVarArray | str,
        name: str,
    ) -> None:
        """Symmetry breaking: array a is lexicographically less than b."""
        a_str = a if isinstance(a, str) else a.name
        b_str = b if isinstance(b, str) else b.name
        self._emit(f"constraint lex_less({a_str}, {b_str}); % {name}")

    def implication(
        self,
        antecedent: MZN_PARAM | tuple[MZN_PARAM, ...],
        consequent: str,
        name: str | None = None,
    ) -> None:
        """
        Post: antecedent -> consequent.
        antecedent is a single boolean expression or a tuple (conjunction).
        consequent is a MiniZinc constraint expression string.
        """
        if not isinstance(antecedent, tuple):
            antecedent = (antecedent,)

        active: list[str] = []
        for var in antecedent:
            if isinstance(var, bool):
                if not var:
                    return
                continue
            if isinstance(var, (int, float)):
                if not var:
                    return
                continue
            active.append(str(var))

        label = f" % {name}" if name else ""

        if not active:
            self._emit(f"constraint {consequent};{label}")
        elif len(active) == 1:
            self._emit(f"constraint {active[0]} -> ({consequent});{label}")
        else:
            conjunction = " /\\ ".join(active)
            self._emit(
                f"constraint ({conjunction}) -> ({consequent});{label}"
            )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def set_objective(self, expr: MZN_PARAM) -> None:
        self._objective_expr = self._to_mzn(expr)

    def max_expr(
        self,
        array: IntVarArray | Sequence[MZN_PARAM],
        name: str,
        lb: int = -10 ** 6,
        ub: int = 10 ** 6,
        minimized: bool = True,
    ) -> IntVar:
        """
        Introduce auxiliary variable z representing max over array.

        If array is an IntVarArray, uses MiniZinc's max() builtin directly
        — exact with no auxiliary lower-bound relaxation.

        If array is a list of mixed MZN_PARAM, uses explicit z >= v
        constraints. When minimized=True the objective drives z to the
        true maximum. When minimized=False z is a lower envelope only.
        """
        aux = self.add_int_var(name, lb=lb, ub=ub)

        if isinstance(array, IntVarArray):
            self._emit(
                f"constraint {aux.name} = max({array.name}); % {name}_max"
            )
        else:
            for i, v in enumerate(array):
                self.add_constraint(
                    f"{aux.name} >= {self._to_mzn(v)}", f"{name}_lb_{i}"
                )

        return aux

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        solver_tag: str | None = None,
        quiet: bool = False,
        time_limit: float | None = None,
        **solver_kwargs: Any,
    ) -> str:
        if solver_tag is None:
            solver_tag = find_available_solver()
            if solver_tag is None:
                raise RuntimeError("No MiniZinc-compatible solver found.")

            logging.info(f"No solver specified, using {solver_tag}.")

        sense = "minimize" if self._minimize else "maximize"
        full_model_str = (
            self._model_str
            + f"\nsolve{self._solve_annotation()} {sense} {self._objective_expr};\n"
        )

        if not quiet:
            logging.debug("MiniZinc model:\n%s", full_model_str)

        model = minizinc.Model()
        model.add_string(full_model_str)

        solver = minizinc.Solver.lookup(solver_tag)
        instance = minizinc.Instance(solver, model)

        timeout = (
            datetime.timedelta(seconds=time_limit)
            if time_limit is not None
            else None
        )

        self._result = instance.solve(
            time_limit=timeout,
            verbose=not quiet,
            **solver_kwargs,
        )

        return str(self._result.status)

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def get_value(self, param: MZN_PARAM) -> float:
        if isinstance(param, (int, float, bool)):
            return float(param)
        if self._result is None:
            raise RuntimeError("Model has not been solved yet.")
        return float(self._result[param])

    def get_int_value(self, var: IntVar | str) -> int:
        if self._result is None:
            raise RuntimeError("Model has not been solved yet.")
        if isinstance(var, int):
            return int(var)

        name = var.name if isinstance(var, IntVar) else var

        if "[" in name and name.endswith("]"):
            base, index = name[:-1].split("[", 1)
            return int(self._result[base][int(index) - 1])

        return int(self._result[name])

    def get_bool_value(self, var: BoolVar | str | bool) -> bool:
        if self._result is None:
            raise RuntimeError("Model has not been solved yet.")
        if isinstance(var, bool):
            return var

        name = var.name if isinstance(var, BoolVar) else var

        if "[" in name and name.endswith("]"):
            base, index = name[:-1].split("[", 1)
            return bool(self._result[base][int(index) - 1])

        return bool(self._result[name])

    def get_int_array(self, array: IntVarArray) -> list[int]:
        if self._result is None:
            raise RuntimeError("Model has not been solved yet.")
        return [int(v) for v in self._result[array.name]]

    def get_bool_array(self, array: BoolVarArray) -> list[bool]:
        if self._result is None:
            raise RuntimeError("Model has not been solved yet.")
        return [bool(v) for v in self._result[array.name]]

    def get_interval_array_values(
        self, intervals: IntervalVarArray
    ) -> list[tuple[int, int, int, bool]]:
        """Returns list of (start, end, duration, present) — 0-indexed."""
        starts = self.get_int_array(intervals.start)
        ends = self.get_int_array(intervals.end)
        durations = self.get_int_array(intervals.duration)
        presents = self.get_bool_array(intervals.present)
        return list(zip(starts, ends, durations, presents))

    def get_objective_value(self) -> float:
        if self._result is None:
            raise RuntimeError("Model has not been solved yet.")
        return float(self._result.objective)

    # ------------------------------------------------------------------
    # Bound queries
    #
    # MiniZinc has no runtime bound inference (no equivalent to Pyomo's
    # compute_bounds_on_expr / FBBT). Bounds on scalars are returned
    # directly. Bounds on variable expressions must be tracked by the
    # subclass at declaration time.
    # ------------------------------------------------------------------

    def get_ub(self, param: MZN_PARAM) -> float:
        if isinstance(param, (int, float, bool)):
            return float(param)
        if isinstance(param, IntVar):
            return float(param.ub)
        raise NotImplementedError(
            f"get_ub cannot infer bound for expression '{param}'. "
            "Track bounds at declaration time in the subclass."
        )

    def get_lb(self, param: MZN_PARAM) -> float:
        if isinstance(param, (int, float, bool)):
            return float(param)
        if isinstance(param, IntVar):
            return float(param.lb)
        raise NotImplementedError(
            f"get_lb cannot infer bound for expression '{param}'. "
            "Track bounds at declaration time in the subclass."
        )

"""
utils.py

Utility model for easy modeling with PuLP.
"""

from typing import Any, Literal, TypeAlias, overload
from collections.abc import Iterable, Mapping, Sequence
from typing_extensions import TypedDict, NotRequired

from pulp import (
    LpProblem,
    lpSum,
    LpVariable,
    LpAffineExpression,
    LpContinuous,
    LpBinary,
    LpConstraint,
)


class SolverConfig(TypedDict, total=False):
    "Configurations for the solver."

    quiet: NotRequired[bool]
    "Whether to suppress solver output."
    time_limit: NotRequired[int]
    "Time limit for the solver in seconds."
    warm_start: NotRequired[bool]
    "Whether to use warm start for the solver."
    keep_files: NotRequired[bool]
    "Whether to keep the solver files after solving."
    options: NotRequired[Sequence[str]]
    "Additional solver options to pass to the solver."
    ...


def parse_solver_config(solver_config: SolverConfig) -> dict[str, Any]:
    config: dict[str, Any] = {}

    if solver_config.pop("quiet", False):
        config["msg"] = 0

    time_limit = solver_config.pop("time_limit", None)
    if time_limit is not None:
        config["timeLimit"] = time_limit

    if solver_config.pop("warm_start", False):
        config["warmStart"] = True

    if solver_config.pop("keep_files", False):
        config["keepFiles"] = True

    return config | solver_config


PULP_EXPRESSION: TypeAlias = LpVariable | LpAffineExpression
PULP_PARAM: TypeAlias = PULP_EXPRESSION | int | float

GLOBAL_BIG_M = (
    1e6  # Default value for big-M method, can be adjusted based on problem scale
)


def get_value(param: PULP_PARAM) -> float | int:
    """
    Get the value of a PULP parameter or expression.

    Args:
        param: A PULP parameter or expression.

    Returns:
        The float value of the parameter or expression.
    """
    if isinstance(param, (int, float)):
        return param

    value = param.value()

    if value is None:
        return 0

    if isinstance(value, (int, float)):
        return value

    raise ValueError(f"Unexpected type: {type(param)}")


def set_initial_value(
    param: PULP_PARAM, value: float | int, check: bool = True
) -> None:
    """
    Set the initial value of a PULP parameter or expression.

    Args:
        param: A PULP parameter or expression.
        value: The value to set.
        check: Whether to check if the value is valid for the parameter type.
    """
    if isinstance(param, LpVariable):
        param.setInitialValue(value, check=check)

    elif isinstance(param, LpAffineExpression):
        # If it's a single variable, set its initial value directly
        if len(param) == 1:
            constant = param.constant
            var, scalar = next(iter(param.items()))

            assert isinstance(var, LpVariable)
            var.setInitialValue((value - constant) / scalar, check=check)


def get_initial_value(param: PULP_PARAM) -> float | int:
    """
    Get the initial value of a PULP parameter or expression.

    Args:
        param: A PULP parameter or expression.

    Returns:
        The initial value of the parameter or expression.
    """
    if isinstance(param, LpVariable):
        return param.varValue if param.varValue is not None else 0

    if isinstance(param, LpAffineExpression):
        # If it's a single variable, get its initial value directly
        value: float | int = param.constant

        for var, scalar in param.items():
            value += get_initial_value(var) * scalar

        return value

    return param


def get_ub(param: PULP_PARAM) -> float | int:
    """
    Get the upper bound of a PULP parameter or expression.

    Args:
        param: A PULP parameter or expression.

    Returns:
        The upper bound of the parameter or expression.
    """
    if isinstance(param, LpVariable):
        ub = param.getUb()
        return ub if ub is not None else float("inf")

    if isinstance(param, LpAffineExpression):
        ub = param.constant
        for var, coeff in param.items():
            ub += coeff * (get_ub(var) if coeff > 0 else get_lb(var))

        assert isinstance(ub, (int, float))
        return ub

    return param


def get_lb(param: PULP_PARAM) -> float | int:
    """
    Get the lower bound of a PULP parameter or expression.

    Args:
        param: A PULP parameter or expression.

    Returns:
        The lower bound of the parameter or expression.
    """
    if isinstance(param, LpVariable):
        lb = param.getLb()
        return lb if lb is not None else 0

    if isinstance(param, LpAffineExpression):
        lb = param.constant
        for var, coeff in param.items():
            lb += coeff * (get_lb(var) if coeff > 0 else get_ub(var))

        assert isinstance(lb, (int, float))
        return lb

    return param


@overload
def get_values(params: PULP_PARAM) -> float | int: ...


@overload
def get_values(params: Sequence[PULP_PARAM]) -> list[float | int]: ...


@overload
def get_values(params: Mapping[str, PULP_PARAM]) -> dict[str, float | int]: ...


def get_values(params: Any) -> Any:
    if isinstance(params, Sequence):
        return [get_value(p) for p in params]

    if isinstance(params, Mapping):
        return {k: get_value(v) for k, v in params.items()}

    return get_value(params)


def is_true(constraint: LpConstraint | bool) -> bool | None:
    if isinstance(constraint, bool):
        return constraint

    if constraint.isNumericalConstant():
        truth: bool = constraint.constant <= 0.0
        return truth

    # If the constraint is not a numerical constant, we cannot determine its truth value
    return None


def pulp_add_constraint(
    model: LpProblem, constraint: LpConstraint | bool, name: str
) -> None:
    """
    Adds a constraint to the PuLP model.

    Parameters:
        model (LpProblem): The PuLP problem instance.
        constraint (LpConstraint | bool): The constraint to add. If it's a boolean, asserts that
        it is True and ignores the addition to the model.
    """
    truth_value = is_true(constraint)

    if truth_value is None:
        assert not isinstance(constraint, bool)

        model.addConstraint(
            constraint,
            name=name,
        )

    elif not truth_value:
        raise ValueError(
            f"Constraint {name} is False, adding it to the model invalidates the model."
        )


global_max_id = 0


def max_pulp(
    model: LpProblem,
    decision_vars: Iterable[PULP_PARAM],
    max_var: LpVariable | None = None,
    cat: str = LpContinuous,
    name: str | None = None,
) -> LpVariable:
    """
    Adds constraints to the model to ensure max_var is at least as large as each decision variable.
    Equivalent to: max_var >= max(decision_vars)

    Parameters:
        model (LpProblem): The PuLP problem instance.
        max_var (LpVariable, optional): The variable to represent the maximum value.
        decision_vars (list[LpVariable | LpAffineExpression | int | float]): The list of variables or expressions.
        name (str, optional): A prefix for constraint names.

    Returns:
        LpVariable: The variable representing the maximum value.
    """
    global global_max_id

    if name is None:
        global_max_id += 1
        name = f"max_var_{global_max_id}"

    if max_var is None:
        max_var = LpVariable(name, lowBound=None, cat=cat)

    for i, var in enumerate(decision_vars):
        model.addConstraint(var <= max_var, f"{name}_ge_var_{i}")

    return max_var


global_and_id = 0


def and_pulp(
    model: LpProblem,
    bin_vars: Iterable[PULP_PARAM],
    and_var: LpVariable | None = None,
    name: str | None = None,
) -> LpVariable | int:
    """
    Adds constraints to the model to ensure and_var is 1 if all bin_vars are 1.
    Equivalent to: and_var == 1 if all(bin_vars) else and_var == 0

    Parameters:
        model (LpProblem): The PuLP problem instance.
        bin_vars (Iterable[LpVariable | int | float]): The binary variables.
        and_var (LpVariable, optional): The variable to represent the AND condition.

    Returns:
        LpVariable: The variable representing the AND condition.
    """
    global global_and_id

    if name is None:
        global_and_id += 1
        name = f"and_var_{global_and_id}"

    if and_var is None:
        and_var = LpVariable(name, lowBound=0, upBound=1, cat=LpBinary)

    count = 0
    sum_vars = LpAffineExpression()
    for var in bin_vars:

        if is_true(var == 1):
            continue

        elif is_true(var == 0):
            return 0

        count += 1
        sum_vars += var

        pulp_add_constraint(
            model,
            and_var <= var,
            f"{name}_le_var_{count}",
        )

    pulp_add_constraint(
        model,
        and_var >= sum_vars - (count - 1),
        f"{name}_ge_sum_vars",
    )

    return and_var


def implication_pulp(
    model: LpProblem,
    antecedent: Iterable[PULP_PARAM] | PULP_PARAM,
    consequent: tuple[PULP_PARAM, Literal["==", "<=", ">="], PULP_PARAM],
    big_m: float = GLOBAL_BIG_M,
    name: str | None = None,
    and_formulation: bool = False,
) -> None:
    """
    Add implication constraints to the model, whenever all antecedent variables are 1,
    the consequent comparison lhs ⋈ rhs must hold. The `big_m` strategy is used to
    model the implication.

    It is equivalent to:
    - if all antecedents == 1, then lhs ⋈ rhs
    - if any antecedent  == 0, then lhs is unconstrained

    Parameters:
        model (LpProblem): The PuLP problem instance.
        antecedent (Iterable[LpVariable] | LpVariable): The antecedent variables.
        consequent (tuple[PULP_EXPRESSION, Literal["==", "<=", ">="], PULP_EXPRESSION | int | float], optional): A tuple containing the lhs, operator, and rhs for the implication. Defaults to None.
        big_m (float, optional): A large constant for the big-M method. Defaults to 1e6.

        When choosing a value for `big_m`, ensure it is large enough to not constrain the model unnecessarily,
        the ideal value is typically upper_bound(lhs) - lower_bound(rhs) if known.
    """
    if isinstance(antecedent, PULP_PARAM):
        antecedent = [antecedent]

    lhs, operator, rhs = consequent
    if all(is_true(premise == 1) for premise in antecedent):

        full_consequent = (
            lhs <= rhs
            if operator == "<="
            else lhs >= rhs if operator == ">=" else lhs == rhs
        )

        pulp_add_constraint(
            model,
            full_consequent,
            name if name is not None else f"{lhs}_{operator}_{rhs}",
        )

        return

    elif any(is_true(premise == 0) for premise in antecedent):
        # If any antecedent is 0, the lhs is unconstrained
        return

    elif and_formulation:
        premise = and_pulp(model, antecedent, name=name)
        n_vars = 1

    else:
        premise = lpSum(antecedent)
        n_vars = sum(1 for _ in antecedent)

    if operator == "==":
        pulp_add_constraint(
            model,
            lhs <= rhs + (n_vars - premise) * big_m,
            f"{name}_le" if name is not None else f"{premise}_{lhs}_le_{rhs}",
        )

        pulp_add_constraint(
            model,
            lhs >= rhs - (n_vars - premise) * big_m,
            f"{name}_ge" if name is not None else f"{premise}_{lhs}_ge_{rhs}",
        )

    elif operator == "<=":
        pulp_add_constraint(
            model,
            lhs <= rhs + (n_vars - premise) * big_m,
            name if name is not None else f"{premise}_{lhs}_le_{rhs}",
        )

    elif operator == ">=":
        pulp_add_constraint(
            model,
            lhs >= rhs - (n_vars - premise) * big_m,
            name if name is not None else f"{premise}_{lhs}_ge_{rhs}",
        )

"""
utils.py

Utility model for easy modeling with PuLP.
"""

from typing import Any, Literal, TypeAlias, overload
from collections.abc import Iterable, Mapping, Sequence

from pulp import (
    LpProblem,
    lpSum,
    LpVariable,
    LpAffineExpression,
    LpContinuous,
    LpBinary,
    LpConstraint,
)

PULP_EXPRESSION: TypeAlias = LpVariable | LpAffineExpression
PULP_PARAM: TypeAlias = PULP_EXPRESSION | int | float

def create_binary_var(name: str, relaxed: bool) -> LpVariable:
    """
    Create a binary variable, or a continuous variable in [0, 1] if relaxed.

    Args:
        name: The name of the variable.
        relaxed: Whether to create a continuous variable in [0, 1] instead of a
        binary variable.

    Returns:
        The created variable.
    """
    return LpVariable(
        name,
        lowBound=0,
        upBound=1,
        cat=LpContinuous if relaxed else LpBinary
    )


def count_variables(variables: Iterable[Any] | PULP_EXPRESSION | int) -> int:
    if isinstance(variables, Mapping):
        return sum(count_variables(v) for v in variables.values())

    if isinstance(variables, Iterable):
        return sum(count_variables(v) for v in variables)

    if isinstance(variables, LpVariable):
        return 1

    return 0


def get_value(param: PULP_PARAM) -> float:
    """
    Get the value of a PULP parameter or expression.

    Args:
        param: A PULP parameter or expression.

    Returns:
        The float value of the parameter or expression.
    """
    if isinstance(param, (int, float)):
        return float(param)

    value = param.value()

    if value is None:
        return 0.0

    return float(value)


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

    elif param != value:
        raise ValueError(
            f"Cannot set initial value of a constant parameter {param} of "
            f"type {type(param)} to {value}."
        )


def get_initial_value(param: PULP_PARAM) -> float:
    """
    Get the initial value of a PULP parameter or expression.

    Args:
        param: A PULP parameter or expression.

    Returns:
        The initial value of the parameter or expression.
    """
    if isinstance(param, LpVariable):
        return param.varValue if param.varValue is not None else 0.0

    if isinstance(param, LpAffineExpression):
        # If it's a single variable, get its initial value directly
        value: float = float(param.constant)

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


def set_ub(param: PULP_PARAM, ub: float | int) -> None:
    """
    Set the upper bound of a PULP parameter.

    Args:
        param: A PULP parameter.
        ub: The upper bound to set.
    """
    if isinstance(param, LpVariable):
        param.upBound = ub

    elif isinstance(param, LpAffineExpression):
        for var, coeff in param.items():
            if coeff > 0:
                set_ub(var, (ub - param.constant) / coeff)
            else:
                set_lb(var, (ub - param.constant) / coeff)

    elif get_ub(param) > ub:
        raise ValueError(
            f"Cannot set upper bound of a constant parameter {param} of "
            f"type {type(param)} to {ub}."
        )

def set_lb(param: PULP_PARAM, lb: float | int) -> None:
    """
    Set the lower bound of a PULP parameter.

    Args:
        param: A PULP parameter.
        lb: The lower bound to set.
    """
    if isinstance(param, LpVariable):
        param.lowBound = lb

    elif isinstance(param, LpAffineExpression):
        for var, coeff in param.items():
            if coeff > 0:
                set_lb(var, (lb - param.constant) / coeff)
            else:
                set_ub(var, (lb - param.constant) / coeff)

    elif get_lb(param) < lb:
        raise ValueError(
            f"Cannot set lower bound of a constant parameter {param} of "
            f"type {type(param)} to {lb}."
        )

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
    name: str | None = None,
    and_formulation: bool = False,
) -> None:
    """
    Add implication constraints to the model, whenever all antecedent variables are 1,
    the consequent comparison lhs ⋈ rhs must hold. The BIG M strategy is used to
    model the implication.

    It is equivalent to:
    - if all antecedents == 1, then lhs ⋈ rhs
    - if any antecedent  == 0, then lhs is unconstrained

    Parameters:
        model (LpProblem): The PuLP problem instance.
        antecedent (Iterable[LpVariable] | LpVariable): The antecedent variables.
        consequent (tuple[PULP_EXPRESSION, Literal["==", "<=", ">="], PULP_EXPRESSION | int | float], optional):
            A tuple containing the lhs, operator, and rhs for the implication. Defaults to None.
    """
    if isinstance(antecedent, PULP_PARAM):
        antecedent = (antecedent,)

    lhs, operator, rhs = consequent

    premises: list[PULP_PARAM] = []
    for var in antecedent:
        if is_true(var == 1):
            continue

        if is_true(var == 0):
            # If any antecedent is 0, the constraint is vacuously satisfied
            return

        premises.append(var)

    if and_formulation:
        premise = and_pulp(model,premises)
        n_vars = 1

    else:
        premise = lpSum(premises)
        n_vars = len(premises)

    if operator == "==":
        pulp_add_constraint(
            model,
            lhs <= rhs + (n_vars - premise) * get_ub(lhs),
            f"{name}_le" if name is not None else f"{premise}_{lhs}_le_{rhs}",
        )

        pulp_add_constraint(
            model,
            lhs >= rhs - (n_vars - premise) * get_lb(lhs),
            f"{name}_ge" if name is not None else f"{premise}_{lhs}_ge_{rhs}",
        )

    elif operator == "<=":
        pulp_add_constraint(
            model,
            lhs <= rhs + (n_vars - premise) * get_ub(lhs),
            name if name is not None else f"{premise}_{lhs}_le_{rhs}",
        )

    elif operator == ">=":
        pulp_add_constraint(
            model,
            lhs >= rhs - (n_vars - premise) * get_lb(lhs),
            name if name is not None else f"{premise}_{lhs}_ge_{rhs}",
        )


global_abs_id = 0


def abs_pulp(
    model: LpProblem,
    expr: PULP_PARAM,
    abs_var: LpVariable | None = None,
    name: str | None = None,
) -> LpVariable:
    """
    Adds constraints to the model to define abs_var as the absolute value of expr.
    Equivalent to: abs_var == |expr|

    Parameters:
        model (LpProblem): The PuLP problem instance.
        expr (LpVariable | LpAffineExpression | int | float): The expression to take the absolute value of.
        abs_var (LpVariable, optional): The variable to represent the absolute value.
        name (str, optional): A prefix for constraint names.

    Returns:
        LpVariable: The variable representing the absolute value.
    """
    global global_abs_id

    if name is None:
        global_abs_id += 1
        name = f"abs_var_{global_abs_id}"

    if abs_var is None:
        abs_var = LpVariable(name, lowBound=0, cat=LpContinuous)

    pulp_add_constraint(
        model,
        abs_var >= expr,
        f"{name}_ge_expr",
    )

    pulp_add_constraint(
        model,
        abs_var >= -expr,
        f"{name}_ge_neg_expr",
    )

    return abs_var


def bilinear_pulp(
    model: LpProblem,
    var1: PULP_PARAM,
    var2: PULP_PARAM,
    bilinear_var: LpVariable | None = None,
    name: str | None = None,
) -> LpVariable:
    """
    Adds constraints to the model to define bilinear_var as the product of var1 and var2.
    This is done using the McCormick relaxation for bilinear terms.

    Parameters:
        model (LpProblem): The PuLP problem instance.
        var1 (LpVariable | LpAffineExpression | int | float): The first variable in the product.
        var2 (LpVariable | LpAffineExpression | int | float): The second variable in the product.
        bilinear_var (LpVariable, optional): The variable to represent the product. Defaults to None.
        name (str, optional): A prefix for constraint names. Defaults to None.

    Returns:
        LpVariable: The variable representing the product of var1 and var2.
    """
    if name is None:
        name = f"bilinear_{var1}_{var2}"

    if bilinear_var is None:
        bilinear_var = LpVariable(name, lowBound=None, cat=LpContinuous)

    # Get bounds for var1 and var2
    var1_lb = get_lb(var1)
    var1_ub = get_ub(var1)
    var2_lb = get_lb(var2)
    var2_ub = get_ub(var2)

    # McCormick relaxation constraints
    pulp_add_constraint(
        model,
        bilinear_var >= var1_lb * var2 + var1 * var2_lb - var1_lb * var2_lb,
        f"{name}_mc1",
    )

    pulp_add_constraint(
        model,
        bilinear_var >= var1_ub * var2 + var1 * var2_ub - var1_ub * var2_ub,
        f"{name}_mc2",
    )

    pulp_add_constraint(
        model,
        bilinear_var <= var1_ub * var2 + var1 * var2_lb - var1_ub * var2_lb,
        f"{name}_mc3",
    )

    pulp_add_constraint(
        model,
        bilinear_var <= var1_lb * var2 + var1 * var2_ub - var1_lb * var2_ub,
        f"{name}_mc4",
    )

    return bilinear_var

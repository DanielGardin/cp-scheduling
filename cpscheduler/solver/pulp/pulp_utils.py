"""
utils.py

Utility model for easy modeling with PuLP.
"""

from typing import Literal, TypeAlias
from collections.abc import Iterable
from typing_extensions import TypedDict, NotRequired

from pulp import (
    LpProblem,
    lpSum,
    LpVariable,
    LpAffineExpression,
    LpContinuous,
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


def implication_pulp(
    model: LpProblem,
    antecedent: Iterable[PULP_PARAM] | PULP_PARAM,
    consequent: tuple[PULP_PARAM, Literal["==", "<=", ">="], PULP_PARAM],
    big_m: float = GLOBAL_BIG_M,
    name: str | None = None,
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

"""
    utils.py

    Utility model for easy modeling with PuLP.
"""
from typing import Literal
from collections.abc import Iterable

from pulp import LpProblem, lpSum, LpVariable, lpDot, LpInteger, LpContinuous, LpAffineExpression

GLOBAL_BIG_M = 1e6  # Default value for big-M method, can be adjusted based on problem scale

global_max_id = 0
def max_pulp(
    model: LpProblem,
    decision_vars: Iterable[LpVariable | LpAffineExpression | int | float],
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
        model.addConstraint(
            var <= max_var,
            f"{name}_ge_var_{i}"
        )

    return max_var

def indicator_constraint(
    model: LpProblem,
    lhs: LpVariable | LpAffineExpression,
    rhs: LpVariable | LpAffineExpression | int | float = 0,
    operator: Literal["==", "<=", ">="] = "==",
    indicators: Iterable[LpVariable] | LpVariable | None = None,
    big_m: float = GLOBAL_BIG_M,
    name: str | None = None
) -> LpVariable:
    """
    Adds constraints to the model to enforce that if all indicator variables are 1, then the comparison must hold.
    It is equivalent to:
    - all indicators == 1 -> lhs ⋈ rhs
    - any indicator  == 0 -> lhs is unconstrained

    Parameters:
        model (LpProblem): The PuLP problem instance.
        lhs (LpVariable | LpAffineExpression): The left-hand side variable or expression.
        rhs (LpVariable | LpAffineExpression | int | float, optional): The right-hand side variable or value. Defaults to 0.
        operator (str, optional): The comparison operator ("==", "<=", ">="). Defaults to "==".
        indicators (Iterable[LpVariable] | LpVariable, optional): The indicator variables. If None, a new variable is created. Defaults to None.
        big_m (float, optional): A large constant for the big-M method. Defaults to 1e6.

        When choosing a value for `big_m`, ensure it is large enough to not constrain the model unnecessarily,
        the ideal value is typically upper_bound(lhs) - lower_bound(rhs) if known.
    """

    if isinstance(indicators, LpVariable):
        indicator = indicators
        n_vars = 1

    elif indicators is None:
        indicator = LpVariable(f"{name}_indicator", lowBound=0, upBound=1, cat=LpInteger)
        n_vars = 1

    elif isinstance(indicators, Iterable):
        indicator = lpSum(indicators)
        n_vars    = sum(1 for _ in indicators)

        ## This implementation seems to be slower for some reason?
        # indicator = LpVariable(name, lowBound=0, upBound=1, cat=LpInteger)
        # n_vars = sum(1 for _ in indicators)

        # model.addConstraint(
        #     indicator >= lpSum(indicators) - n_vars + 1
        # )

        # for i, ind in enumerate(indicators):
        #     model.addConstraint(
        #         indicator <= ind,
        #         f"{name}_indicator_{i}"
        #     )

    if operator == "==":
        model.addConstraint(
            lhs <= rhs + (n_vars - indicator) * big_m,
            name if name is not None else f"{indicator}_{lhs}_le_{rhs}"
        )

        model.addConstraint(
            lhs >= rhs - (n_vars - indicator) * big_m,
            name if name is not None else f"{indicator}_{lhs}_ge_{rhs}"
        )

    elif operator == "<=":
        model.addConstraint(
            lhs <= rhs + (n_vars - indicator) * big_m,
            name if name is not None else f"{indicator}_{lhs}_le_{rhs}"
        )

    elif operator == ">=":
        model.addConstraint(
            lhs >= rhs - (n_vars - indicator) * big_m,
            name if name is not None else f"{indicator}_{lhs}_ge_{rhs}"
        )

    return indicator
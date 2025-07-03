"""
utils.py

Utility model for easy modeling with PuLP.
"""

from typing import Literal
from collections.abc import Iterable

from pulp import LpProblem, lpSum, LpVariable, LpContinuous

from .common import PULP_EXPRESSION, PULP_PARAM

GLOBAL_BIG_M = (
    1e6  # Default value for big-M method, can be adjusted based on problem scale
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
    antecedent: Iterable[LpVariable] | LpVariable,
    consequent: tuple[PULP_EXPRESSION, Literal["==", "<=", ">="], PULP_PARAM],
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
    if isinstance(antecedent, LpVariable):
        antecedent = [antecedent]

    premise = lpSum(antecedent)
    n_vars = sum(1 for _ in antecedent)

    lhs, operator, rhs = consequent

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
            lhs <= rhs + (n_vars - premise) * big_m,
            f"{name}_le" if name is not None else f"{premise}_{lhs}_le_{rhs}",
        )

        model.addConstraint(
            lhs >= rhs - (n_vars - premise) * big_m,
            f"{name}_ge" if name is not None else f"{premise}_{lhs}_ge_{rhs}",
        )

    elif operator == "<=":
        model.addConstraint(
            lhs <= rhs + (n_vars - premise) * big_m,
            name if name is not None else f"{premise}_{lhs}_le_{rhs}",
        )

    elif operator == ">=":
        model.addConstraint(
            lhs >= rhs - (n_vars - premise) * big_m,
            name if name is not None else f"{premise}_{lhs}_ge_{rhs}",
        )

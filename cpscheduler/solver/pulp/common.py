from typing import TypeAlias

from pulp import LpVariable, LpAffineExpression

PULP_EXPRESSION: TypeAlias = LpVariable | LpAffineExpression
PULP_PARAM: TypeAlias = PULP_EXPRESSION | int | float

import ast
from typing import Final, Literal, TypeAlias

from cpscheduler.environment.constants import EzPickle

BuiltinSymbols = Literal["n_tasks", "n_jobs", "n_machines"]
BaseShapeDim = int | BuiltinSymbols | str | None
"""Symbolic dimensions for shapes.

Allows building features with dynamic shaping, such as ("n_tasks", 2).
Symbolic dimensions are resolved in runtime.
"""

ShapeDim: TypeAlias = "BaseShapeDim | SymbolicDim"

# FUTURE: Extend this to user-defined symbols
VALID_SYMBOLS: frozenset[BuiltinSymbols] = frozenset(
    ["n_tasks", "n_jobs", "n_machines"]
)


def _parse_affine_expr(expr: str) -> dict[str, int]:
    """
    Parse a string affine expression into a coefficient dict.

    Returns keys for all symbols, with 0 for missing symbols.
    Always includes an "const" key for the constant term.
    Raises ValueError on any invalid or unsupported expression.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")

    except SyntaxError as e:
        raise ValueError(f"Invalid expression: '{expr}'.") from e

    result: dict[str, int] = {symbol: 0 for symbol in VALID_SYMBOLS}
    result["const"] = 0

    _visit(tree.body, result, sign=1)

    return result


def _visit(node: ast.expr, result: dict[str, int], sign: int) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int):
            raise ValueError(
                f"Only integer constants are supported, got: {node.value!r}."
            )

        result["const"] += sign * node.value

    elif isinstance(node, ast.Name):
        if node.id not in VALID_SYMBOLS:
            raise ValueError(
                f"Unknown symbol: '{node.id}'. Must be one of {VALID_SYMBOLS}."
            )

        result[node.id] += sign

    elif isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            _visit(node.left, result, sign)
            _visit(node.right, result, sign)

        elif isinstance(node.op, ast.Sub):
            _visit(node.left, result, sign)
            _visit(node.right, result, -sign)

        elif isinstance(node.op, ast.Mult):
            _visit_mult(node.left, node.right, result, sign)

        else:
            raise ValueError(
                f"Unsupported operator: {type(node.op).__name__}. "
                "Only +, -, * are allowed."
            )

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        _visit(node.operand, result, -sign)

    else:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}.")


def _visit_mult(
    left: ast.expr,
    right: ast.expr,
    result: dict[str, int],
    sign: int,
) -> None:
    # Exactly one side must be a constant int, the other a known symbol.
    if isinstance(left, ast.Constant) and isinstance(right, ast.Name):
        coef, sym_node = left, right

    elif isinstance(left, ast.Name) and isinstance(right, ast.Constant):
        coef, sym_node = right, left

    else:
        expr = ast.dump(ast.BinOp(left=left, op=ast.Mult(), right=right))

        raise ValueError(
            "Multiplication is only supported between an integer constant and a symbol "
            f"(e.g. '2*n_tasks'). Got: {expr}."
        )

    if not isinstance(coef.value, int):
        raise ValueError(
            f"Coefficient must be an integer, got: {coef.value!r}."
        )

    if sym_node.id not in VALID_SYMBOLS:
        raise ValueError(
            f"Unknown symbol: '{sym_node.id}'. Must be one of {VALID_SYMBOLS}."
        )

    result[sym_node.id] += sign * coef.value


class SymbolicDim(EzPickle):
    _coefs: dict[str, int]
    _const_value: int

    @property
    def const(self) -> int:
        return self._const_value

    @property
    def coefs(self) -> dict[str, int]:
        return {symbol: coef for symbol, coef in self._coefs.items()}

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(self._coefs.keys())

    def __init__(
        self,
        const: int = 0,
        *,
        n_tasks: int = 0,
        n_jobs: int = 0,
        n_machines: int = 0,
        **symbol_values: int,
    ) -> None:
        self._const_value = const
        coefs = {
            "n_tasks": n_tasks,
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            **symbol_values,
        }

        self._coefs = {sym: coef for sym, coef in coefs.items() if coef != 0}

    @classmethod
    def from_shapedim(cls, dim: int | str) -> "SymbolicDim":
        if isinstance(dim, int):
            return cls(const=dim)

        # Logic for string symbolic dimensions (e.g. "n_tasks + 2")
        coefs = _parse_affine_expr(dim)

        return cls(**coefs)

    @property
    def raw(self) -> BaseShapeDim:
        if self.is_constant():
            return self._const_value

        return str(self)

    def is_constant(self) -> bool:
        return not self._coefs

    def is_symbolic(self) -> bool:
        return not self.is_constant()

    def resolve(self, **symbol_values: int) -> int:
        value = self._const_value

        for sym, coef in self._coefs.items():
            if sym not in symbol_values:
                raise ValueError(
                    f"Cannot resolve SymbolicDim: missing value for symbol '{sym}'."
                )

            value += coef * symbol_values[sym]

        return value

    def __add__(self, other: ShapeDim) -> "SymbolicDim":
        if other is None:
            raise ValueError(
                "Cannot sum a symbolic dim with variadic None dimension."
            )

        if not isinstance(other, SymbolicDim):
            other = self.from_shapedim(other)

        symbols = set(self._coefs) | set(other._coefs)
        return SymbolicDim(
            const=self._const_value + other._const_value,
            **{
                sym: self._coefs.get(sym, 0) + other._coefs.get(sym, 0)
                for sym in symbols
            },
        )

    def __radd__(self, other: ShapeDim) -> "SymbolicDim":
        return self + other

    def __sub__(self, other: ShapeDim) -> "SymbolicDim":
        if other is None:
            raise ValueError(
                "Cannot subtract a symbolic dim with variadic None dimension."
            )

        if not isinstance(other, SymbolicDim):
            other = self.from_shapedim(other)

        symbols = set(self._coefs) | set(other._coefs)
        return SymbolicDim(
            const=self._const_value - other._const_value,
            **{
                sym: self._coefs.get(sym, 0) - other._coefs.get(sym, 0)
                for sym in symbols
            },
        )

    def __rsub__(self, other: ShapeDim) -> "SymbolicDim":
        return (-self) + other

    def __mul__(self, other: int) -> "SymbolicDim":
        return SymbolicDim(
            const=self._const_value * other,
            **{sym: coef * other for sym, coef in self._coefs.items()},
        )

    def __rmul__(self, other: int) -> "SymbolicDim":
        return self * other

    def __neg__(self) -> "SymbolicDim":
        return SymbolicDim(
            const=-self._const_value,
            **{sym: -coef for sym, coef in self._coefs.items()},
        )

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, SymbolicDim):
            if not isinstance(value, int | str):
                return NotImplemented

            value = self.from_shapedim(value)

        return (
            self._coefs == value._coefs
            and self._const_value == value._const_value
        )

    def __hash__(self) -> int:
        return hash((self._coefs, self._const_value))

    def __repr__(self) -> str:
        parts: list[str] = []

        for sym, coef in self._coefs.items():
            if coef == 1:
                parts.append(sym)

            elif coef == -1:
                parts.append(f"-{sym}")

            else:
                parts.append(f"{coef}*{sym}")

        if self._const_value != 0 or not parts:
            parts.append(str(self._const_value))

        result = parts[0]
        for part in parts[1:]:
            if part.startswith("-"):
                result += f" - {part[1:]}"
            else:
                result += f" + {part}"

        return result


N_TASKS: Final = SymbolicDim(n_tasks=1)
N_JOBS: Final = SymbolicDim(n_jobs=1)
N_MACHINES: Final = SymbolicDim(n_machines=1)

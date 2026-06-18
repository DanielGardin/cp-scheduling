"""Symbolic algebra for features with variable shape dimensions.

This module introduces string-based symbolic dimensions for static specification
of features with dynamic shaping, such as ("n_tasks", 2).
Affine expressions are also allowed, where shapes like ("n_tasks", "2*n_jobs+1")
are allowed, and parsed as SymbolicDim objects.
Symbol values are resolved during runtime.
"""

from __future__ import annotations

import ast
from typing import Literal, TypeAlias, overload

from cpscheduler.environment.constants import EzPickle

BuiltinSymbols = Literal["n_tasks", "n_jobs", "n_machines"]
BaseShapeDim = int | BuiltinSymbols | str | None

ShapeDim: TypeAlias = "BaseShapeDim | SymbolicDim"


REQUIRED_SYMBOLS: frozenset[str] = frozenset(
    ["const", "n_tasks", "n_jobs", "n_machines"]
)


def _parse_affine_expr(expr: str) -> dict[str, int]:
    """Parse a string affine expression into a coefficient dict.

    Returns keys for all valid symbols, with 0 for missing symbols.
    Always includes an "const" key for the constant term.

    Raises ValueError on any invalid or unsupported expression.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")

    except SyntaxError as e:
        raise ValueError(f"Invalid expression: '{expr}'.") from e

    result: dict[str, int] = {"const": 0}
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
        result.setdefault(node.id, 0)
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

    result.setdefault(sym_node.id, 0)
    result[sym_node.id] += sign * coef.value


class SymbolicDim(EzPickle):
    """Represents a symbolic shape dimension as an affine expression of known symbols.

    For example, "2*n_tasks + 3" would be represented as a SymbolicDim with
    coefs {"n_tasks": 2} and const 3.
    """

    _coefs: dict[str, int]
    _const_value: int

    def __init__(
        self,
        const: int = 0,
        *,
        n_tasks: int = 0,
        n_jobs: int = 0,
        n_machines: int = 0,
        **symbol_values: int,
    ) -> None:
        """Initialize a SymbolicDim with a constant term and symbol coefficients."""
        self._const_value = const
        coefs = {
            "n_tasks": n_tasks,
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            **symbol_values,
        }

        self._coefs = {sym: coef for sym, coef in coefs.items() if coef != 0}

    @classmethod
    def from_shapedim(cls, dim: int | str) -> SymbolicDim:
        """Create a SymbolicDim from a dimension specification, parsing strings as affine expressions."""
        if isinstance(dim, int):
            return cls(const=dim)

        # Logic for string symbolic dimensions (e.g. "n_tasks + 2")
        coefs = _parse_affine_expr(dim)

        return cls(**coefs)

    @property
    def raw(self) -> BaseShapeDim:
        """The raw dimension specification, as provided by the user."""
        if self.is_constant():
            return self._const_value

        return str(self)

    @property
    def const(self) -> int:
        """The constant term of the symbolic dimension."""
        return self._const_value

    @property
    def coefs(self) -> dict[str, int]:
        """The symbol coefficients of the symbolic dimension."""
        return dict(self._coefs.items())

    @property
    def symbols(self) -> frozenset[str]:
        """The set of symbols involved in the symbolic dimension."""
        return frozenset(self._coefs.keys())

    @property
    def n_symbols(self) -> int:
        """The number of symbols involved in the symbolic dimension."""
        return len(self._coefs)

    def is_constant(self) -> bool:
        """Whether this symbolic dimension is actually a constant (i.e. has no symbols)."""
        return not self._coefs

    def is_atomic(self) -> bool:
        """Whether this symbolic dimension is atomic (i.e. of the form 'n_tasks' with coef 1)."""
        return len(self._coefs) == 1 and next(iter(self._coefs.values())) == 1

    def solve_symbol(self, value: int) -> dict[str, int]:
        """Solve for the value of the single symbol in this dimension, given the total value."""
        if self.is_constant():
            raise ValueError("Cannot solve a constant dimension.")

        if self.n_symbols > 1:
            raise ValueError(
                f"Cannot solve a symbolic dimension for {self} "
                "because it has multiple symbols."
            )

        sym, coef = next(iter(self._coefs.items()))

        if coef == 0:
            raise ValueError(
                "Invalid symbolic dimension with zero coefficient."
            )

        div, mod = divmod(value - self._const_value, coef)

        if mod != 0:
            raise ValueError(
                f"Cannot solve for symbol '{sym}' in {self} with value {value}. "
                "The value does not satisfy the affine equation."
            )

        return {sym: div}

    def resolve(self, **symbol_values: int) -> int:
        """Resolve the symbolic dimension to an integer value, given values for the symbols."""
        value = self._const_value

        for sym, coef in self._coefs.items():
            if sym not in symbol_values:
                raise ValueError(
                    f"Cannot resolve SymbolicDim: missing value for symbol '{sym}'."
                )

            value += coef * symbol_values[sym]

        return value

    def __add__(self, other: ShapeDim) -> SymbolicDim:
        """Add another shape dimension to this symbolic dimension, returning a new SymbolicDim."""
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

    def __radd__(self, other: ShapeDim) -> SymbolicDim:
        """Right-add another shape dimension to this symbolic dimension, returning a new SymbolicDim."""
        return self + other

    def __sub__(self, other: ShapeDim) -> SymbolicDim:
        """Subtract another shape dimension from this symbolic dimension, returning a new SymbolicDim."""
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

    def __rsub__(self, other: ShapeDim) -> SymbolicDim:
        """Right-subtract another shape dimension from this symbolic dimension, returning a new SymbolicDim."""
        return (-self) + other

    def __mul__(self, other: int) -> SymbolicDim:
        """Multiply this symbolic dimension by an integer constant, returning a new SymbolicDim."""
        return SymbolicDim(
            const=self._const_value * other,
            **{sym: coef * other for sym, coef in self._coefs.items()},
        )

    def __rmul__(self, other: int) -> SymbolicDim:
        """Right-multiply this symbolic dimension by an integer constant, returning a new SymbolicDim."""
        return self * other

    def __neg__(self) -> SymbolicDim:
        """Negate this symbolic dimension, returning a new SymbolicDim."""
        return SymbolicDim(
            const=-self._const_value,
            **{sym: -coef for sym, coef in self._coefs.items()},
        )

    def __eq__(self, value: object, /) -> bool:
        """Check equality with another symbolic dimension or shape dim specification."""
        if isinstance(value, SymbolicDim):
            return (
                self._coefs == value._coefs
                and self._const_value == value._const_value
            )

        if isinstance(value, int):
            return self.is_constant() and self._const_value == value

        if isinstance(value, str):
            try:
                return self == self.from_shapedim(value)

            except ValueError:
                return False

        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on the coefficients and constant value."""
        return hash((tuple(self._coefs.items()), self._const_value))

    def __repr__(self) -> str:
        """Return a string representation of the symbolic dimension, reconstructing the affine expression."""
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


@overload
def symbolic_shape(
    raw_shape: tuple[BaseShapeDim, ...],
) -> tuple[SymbolicDim | None, ...]: ...


@overload
def symbolic_shape(raw_shape: None) -> None: ...


def symbolic_shape(
    raw_shape: tuple[BaseShapeDim, ...] | None,
) -> tuple[SymbolicDim | None, ...] | None:
    """Turn a shape object into a tuple of SymbolicDims."""
    if raw_shape is None:
        return None

    return tuple(
        SymbolicDim.from_shapedim(dim) if dim is not None else None
        for dim in raw_shape
    )


@overload
def resolve_shape(
    shape: tuple[SymbolicDim | None, ...],
) -> tuple[int | None, ...]: ...


@overload
def resolve_shape(shape: None) -> None: ...


def resolve_shape(
    shape: tuple[SymbolicDim | None, ...] | None, **symbol_values: int
) -> tuple[int | None, ...] | None:
    """Materizalize the shape defined by SimbolicDims."""
    if shape is None:
        return None

    return tuple(
        dim.resolve(**symbol_values) if isinstance(dim, SymbolicDim) else None
        for dim in shape
    )

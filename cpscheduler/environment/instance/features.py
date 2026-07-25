"""Feature classes for scheduling instance specifications and data management."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Generic, Literal

from typing_extensions import TypeIs, TypeVar

from cpscheduler.environment.constants import EzPickle, Singleton, hash_anything
from cpscheduler.environment.specs.feature_spec import FeatureViewSpec
from cpscheduler.environment.utils.symbols import (
    BaseShapeDim,
    SymbolicDim,
    resolve_shape,
    solve_shape,
    symbolic_shape,
    to_raw_shape,
)


# This is used to distinguish between features that have no data loaded and those
# that have data loaded with a value of None or other falsy values.
class _UnsetType(Singleton):
    """A singleton type to represent unset values for feature data."""


UNSET = _UnsetType()


def is_unset(value: object) -> TypeIs[_UnsetType]:
    """Check if a value is the UNSET singleton."""
    return value is UNSET


ValueType = Literal[
    # Numeric
    "continuous",  # Real-valued, unbounded
    "discrete",  # Integer-valued, unbounded
    "binary",  # Boolean-valued, {0, 1}
    "count",  # Non-negative integer-valued
    # Bounded numeric
    "normalized",  # Real-valued, bounded in [low, high]
    "probability",  # Real-valued, bounded in [0, 1], sum to 1
    # Scheduling quantities
    "time",  # Integer-valued, bounded in [MIN_TIME, MAX_TIME]
    "duration",  # Non-negative integer-valued time duration
    "cost",  # Non-negative real-valued cost
    # Identifiers
    "id",  # Integer-valued, unique identifier
    "task_id",  # Integer-valued, bounded in [0, n_tasks)
    "job_id",  # Integer-valued, bounded in [0, n_jobs)
    "machine_id",  # Integer-valued, bounded in [0, n_machines)
    # Ordered / categorical
    "order",  # Non-negative integer-valued, teorically bounded
    "categorical",  # Categorical, integer-valued, bounded in [0, n_categories)
    # Unknown
    "unknown",  # Non-structured, or non-scalar
]


class FeatureMetadata(EzPickle):
    """Metadata for a scheduling instance feature."""

    value_type: ValueType
    shape: tuple[SymbolicDim | None, ...] | None

    n_categories: int | None
    low: float | None
    high: float | None

    def __init__(
        self,
        value_type: ValueType,
        shape: tuple[BaseShapeDim, ...] | None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize feature metadata.

        Parameters
        ----------
        value_type: ValueType
            The type of values the feature holds, used for validation and interpretation.

        shape: tuple[BaseShapeDim, ...] or None
            The shape of the feature data, where BaseShapeDim can be an int or a
            symbolic dimension. If None, the shape is not specified.

        n_categories: int | None, default None
            The number of categories, if the feature is categorical.

        low: float | None, default None
            The lower bound of the feature values, if applicable.

        high: float | None, default None
            The upper bound of the feature values, if applicable.

        """
        self.value_type = value_type
        self.shape = symbolic_shape(shape)
        self.n_categories = n_categories
        self.low = low
        self.high = high

    @property
    def symbols(self) -> set[str]:
        """Return the set of symbols used in the feature's shape."""
        if self.shape is None:
            return set()

        symbols: set[str] = set()
        for dim in self.shape:
            if isinstance(dim, SymbolicDim):
                symbols.update(dim.symbols)

        return symbols

    @property
    def raw_shape(self) -> tuple[BaseShapeDim, ...] | None:
        """Return the unresolved shape."""
        return to_raw_shape(self.shape)

    def resolve_shape(self, **symbols: int) -> tuple[int | None, ...] | None:
        """Resolve the symbolic dimensions in the feature's shape to concrete integers."""
        return resolve_shape(self.shape, **symbols)


_T = TypeVar("_T", default=Any)


class Feature(EzPickle, Generic[_T]):
    """Storage class for a scheduling instance feature.

    This class serves two purposes:
    - It defines a component requirement for a scheduling instance.
    - It manages the data associated with that feature, whether it is fixed
    or loaded at runtime.
    """

    name: str

    _preprocess: Callable[[Any], _T] | None
    _storage: _T | _UnsetType  # Persistent data: owner == _storage is not UNSET
    _data: _T | _UnsetType  # Current data
    owner: bool

    metadata: FeatureMetadata
    view: FeatureViewSpec[_T, Any]

    def __init__(
        self,
        name: str,
        optional: bool = False,
        preprocess: Callable[[Any], _T] | None = None,
        view: FeatureViewSpec[_T, Any] | None = None,
        owner: bool = False,
        *,
        value_type: ValueType = "unknown",
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize a feature with the given parameters.

        Parameters
        ----------
        name: str
            The name of the feature.

        optional: bool, optional
            Whether the feature is optional. If True, the feature can be left unset
            without causing errors. Default is False.

        default: _T or UNSET, optional
            The default value of the feature. If not provided, it is set to UNSET,
            indicating that the feature has no default data.

        preprocess: Callable[[Any], _T] or None, optional
            A function to preprocess the feature data before it is stored or used.
            If None, no preprocessing is applied. Default is None.

        view: FeatureViewSpec[_T, Any] or None, optional
            A specification of how the feature data should be viewed or interpreted.
            If None, a default view is created based on the shape and additional
            parameters. Default is None.

        owner: bool, optional
            Whether this feature instance owns its data. If True, the feature is
            a provider, and must call `own_data` to set its persistent data.
            If False, the feature is a consumer and can load data from other features.

        Metadata
        -----------------

        value_type: ValueType, optional
            The type of values the feature holds, used for validation and interpretation.
            Default is "unknown".

        shape: tuple[BaseShapeDim, ...] or None, optional
            The shape of the feature data, where BaseShapeDim can be an int or a
            symbolic dimension. If None, the shape is not specified. Default is None.

        n_categories: int or None, optional
            The number of categories for categorical features. If None, the feature
            is not treated as categorical. Default is None.

        low: float or None, optional
            The lower bound for the feature values, used for validation. If None,
            no lower bound is enforced. Default is None.

        high: float or None, optional
            The upper bound for the feature values, used for validation. If None,
            no upper bound is enforced. Default is None.

        """
        self.name = name
        self.optional = optional

        self._preprocess = preprocess
        self._storage = UNSET
        self._data = UNSET

        self.owner = owner

        self.metadata = FeatureMetadata(
            value_type=value_type,
            shape=shape,
            n_categories=n_categories,
            low=low,
            high=high,
        )

        self.view = (
            view
            if view is not None
            else FeatureViewSpec(
                value_type=value_type,
                shape=shape,
                n_categories=n_categories,
                low=low,
                high=high,
            )
        )

    @property
    def loaded(self) -> bool:
        """Check if the feature has loaded data."""
        return self._data is not UNSET

    @property
    def value(self) -> _T:
        """Get the feature's loaded data."""
        if not is_unset(self._data):
            return self._data

        raise ValueError(f"Feature {self.name} has no loaded data.")

    @property
    def shape(self) -> tuple[SymbolicDim | None, ...] | None:
        """Return the symbolic shape of the feature."""
        return self.metadata.shape

    @property
    def symbols(self) -> set[str]:
        """Return the set of symbols used in the feature's shape."""
        return self.metadata.symbols

    def reset(self) -> None:
        """Overwrite feature's data with its persistent value."""
        self._data = deepcopy(self._storage)

    def own_data(self, data: Any) -> None:
        """Overwrite the feature's persistent data."""
        _data = self._preprocess(data) if self._preprocess is not None else data

        self._storage = _data
        self._data = deepcopy(_data)
        self.owner = True

    def load_data(self, data: _T) -> None:
        """Set the feature's current data."""
        if self.owner:
            raise RuntimeError(
                f"Cannot load data for feature '{self.name}', it is an owner."
            )

        if self.loaded:
            raise RuntimeError(
                f"Feature '{self.name}' already has loaded data: {self._data}."
            )

        self._data = (
            self._preprocess(data) if self._preprocess is not None else data
        )

    def shared_data(self, source: "Feature[_T]") -> None:
        """Get data from another feature, sharing the same data reference.

        The source feature must be compatible in terms of specification
        and must be an owner of its data.
        The current feature must not be an owner, as it will become a consumer
        sharing the source's data.
        """
        # TODO: Check if the source feature is compatible in terms of specification

        if not source.owner:
            raise RuntimeError(
                f"Cannot share source feature '{source.name}' data, "
                "it is not an owner of its data."
            )

        if self.owner:
            raise RuntimeError(
                f"Cannot gather data from other feature, feature '{self.name}' "
                "is an owner."
            )

        if is_unset(source._data):
            raise RuntimeError(
                f"Source feature '{source.name}' has no loaded data to share."
            )

        self._data = source._data

    def resolve_shape(self, **symbols: int) -> tuple[int | None, ...] | None:
        """Resolve the symbolic dimensions in the feature's shape to concrete integers."""
        return self.metadata.resolve_shape(**symbols)

    def solve_symbols(self) -> dict[str, int]:
        """Solve the symbolic dimensions in the feature's shape to concrete integers."""
        if not self.loaded:
            raise ValueError(
                f"Feature {self.name} has no loaded data to solve symbols."
            )

        shape = self.metadata.shape

        if shape is None:
            return {}

        symbols = solve_shape(shape, self._data)

        return symbols or {}

    def validate(self) -> None:
        """Validate the feature's loaded data against its specification."""
        if not self.loaded and not self.optional:
            raise RuntimeError(
                f"Feature {self.name} is required but has no loaded data."
            )

    def compute_hash(self) -> int:
        """Compute a hash for the feature data."""
        if not self.loaded:
            raise RuntimeError(
                f"Feature {self.name} has no loaded data to compute hash."
            )

        return hash_anything(self._data)

    def materialize(self, spec: FeatureViewSpec[_T, Any] | None = None) -> Any:
        """Return a materialized representation of the feature's data."""
        if spec is None:
            spec = self.view

        return spec.materialize(self.value)

    def __eq__(self, value: object, /) -> bool:
        """Check equality of features based on their name and specification."""
        return isinstance(value, Feature) and self.name == value.name

    def __repr__(self) -> str:
        """Return a string representation of the feature."""
        attrs = [
            f"name={self.name!r}",
            f"owner={self.owner}",
            f"loaded={self.loaded}",
        ]

        if self.optional:
            attrs.append("optional=True")

        return f"Feature({', '.join(attrs)})"

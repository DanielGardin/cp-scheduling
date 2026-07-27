"""Feature classes for scheduling instance specifications and data management."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Generic, override

from mypy_extensions import mypyc_attr
from typing_extensions import TypeIs, TypeVar

from cpscheduler.environment.constants import EzPickle, Singleton, hash_anything
from cpscheduler.environment.instance.metadata import FeatureMetadata, ValueType
from cpscheduler.environment.specs.feature_spec import (
    AdjacencyMatrixViewSpec,
    FeatureViewSpec,
    from_metadata,
)
from cpscheduler.environment.utils.symbols import BaseShapeDim, SymbolicDim


# This is used to distinguish between features that have no data loaded and those
# that have data loaded with a value of None or other falsy values.
class _UnsetType(Singleton):
    """A singleton type to represent unset values for feature data."""


UNSET = _UnsetType()


def is_unset(value: object) -> TypeIs[_UnsetType]:
    """Check if a value is the UNSET singleton."""
    return value is UNSET


_T = TypeVar("_T", default=Any)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
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

    def __init__(
        self,
        name: str,
        optional: bool = False,
        preprocess: Callable[[Any], _T] | None = None,
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

    # Metadata-related methods
    @property
    def shape(self) -> tuple[SymbolicDim | None, ...] | None:
        """Return the symbolic shape of the feature."""
        return self.metadata.shape

    @property
    def symbols(self) -> set[str]:
        """Return the set of symbols used in the feature's shape."""
        return self.metadata.symbols

    def resolve_shape(self, **symbols: int) -> tuple[int | None, ...] | None:
        """Resolve the symbolic dimensions in the feature's shape to concrete integers."""
        return self.metadata.resolve_shape(**symbols)

    def solve_symbols(self) -> dict[str, int]:
        """Solve the symbolic dimensions in the feature's shape to concrete integers."""
        if not self.loaded:
            raise ValueError(
                f"Feature {self.name} has no loaded data to solve symbols."
            )

        return self.metadata.solve_symbols(self._data)

    # Data management methods
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

    def reset(self) -> None:
        """Overwrite feature's data with its persistent value."""
        self._data = deepcopy(self._storage)

    def empty(self) -> None:
        """Clear the feature's loaded data, setting it to UNSET."""
        self._data = UNSET

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
        if not source.metadata.is_compatible(self.metadata):
            raise ValueError(
                f"Cannot share data from feature '{source.name}' to feature "
                f"'{self.name}', specifications are not compatible."
            )

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

    # Observation and materialization methods
    def possible_views(self) -> dict[str, FeatureViewSpec[_T, Any]]:
        """Return a list of possible views for the feature's data."""
        return {"default": from_metadata(self.metadata)}

    def materialize(
        self, spec: FeatureViewSpec[_T, Any] | None = None, **symbols: int
    ) -> Any:
        """Return a materialized representation of the feature's data."""
        if spec is None:
            spec = self.possible_views()["default"]

        return spec.materialize(self.value, **symbols)


# Specialized feature classes for specific data types


class DAGFeature(Feature[dict[int, list[int]]]):
    """Feature class for representing Directed Acyclic Graphs (DAGs)."""

    n_nodes: SymbolicDim

    def __init__(
        self,
        name: str,
        n_nodes: int | str,
        optional: bool = False,
        preprocess: Callable[[Any], dict[int, list[int]]] | None = None,
        owner: bool = False,
        *,
        value_type: ValueType = "unknown",
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize a DAG feature with the given parameters."""
        super().__init__(
            name=name,
            optional=optional,
            preprocess=preprocess,
            owner=owner,
            value_type=value_type,
            shape=None,
            n_categories=n_categories,
            low=low,
            high=high,
        )

        self.n_nodes = SymbolicDim.from_shapedim(n_nodes)

    @override
    def possible_views(
        self,
    ) -> dict[str, FeatureViewSpec[dict[int, list[int]], Any]]:
        return {
            "default": from_metadata(self.metadata),
            "adjacency_matrix": AdjacencyMatrixViewSpec(
                value_type=self.metadata.value_type,
                n_nodes=self.n_nodes.raw,
                n_categories=self.metadata.n_categories,
                low=self.metadata.low,
                high=self.metadata.high,
            ),
        }

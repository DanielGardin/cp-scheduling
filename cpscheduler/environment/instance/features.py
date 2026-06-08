"""Feature classes for scheduling instance specifications and data management."""

from collections.abc import Sequence
from typing import Any, Generic, Literal

from typing_extensions import Self, TypeIs, TypeVar

from cpscheduler.environment.constants import EzPickle, Singleton
from cpscheduler.environment.specs.feature_spec import (
    FeatureSpec,
    Scope,
    SemanticType,
)
from cpscheduler.environment.specs.symbols import BaseShapeDim


# This is used to distinguish between features that have no data loaded and those
# that have data loaded with a value of None or other falsy values.
class _UnsetType(Singleton):
    """A singleton type to represent unset values for feature data."""


UNSET = _UnsetType()


def is_unset(value: object) -> TypeIs[_UnsetType]:
    """Check if a value is the UNSET singleton."""
    return value is UNSET


def has_shape(data: Any, shape: tuple[int | None, ...]) -> bool:
    """Check if the data has the specified shape.

    If data implements the `shape` attribute, it is used to check the shape.
    Otherwise, recursive checks are performed for sequences, treating None
    in the shape as a wildcard that matches any size in that dimension.
    """
    if hasattr(data, "shape"):
        return tuple(data.shape) == shape

    if shape and shape[0] is None:
        return True

    if isinstance(data, Sequence) and not isinstance(data, str):
        if len(data) == 0:
            return True

        if len(shape) == 0:
            return False

        if len(data) != shape[0]:
            return False

        return has_shape(data[0], shape[1:])

    return len(shape) == 0


_T = TypeVar("_T", default=Any)


class Feature(EzPickle, Generic[_T]):
    """Base class for features of scheduling instances.

    A feature represents a specific aspect of the scheduling instance, such as
    the processing times of tasks, the due dates of jobs, or the availability of
    machines.
    Each feature has a name, a specification, and can optionally own its data.

    Features can be shared between different components of the environment, and they
    can be dynamic, meaning that their data can change during the scheduling process.
    """

    name: str
    spec: FeatureSpec
    optional: bool
    owner: bool

    _storage: _T | _UnsetType
    _data: _T | _UnsetType

    dynamic: bool

    def __init__(
        self,
        name: str,
        scope: Scope,
        semantic: SemanticType,
        *,
        optional: bool = False,
        default: _T | _UnsetType = UNSET,
        owner: bool | None = None,
        dynamic: bool = False,
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

        scope: {"task", "job", "machine", "global"}
            The scope of the feature, indicating whether it applies to tasks, jobs,
            machines, or is global to the instance.

        semantic: str
            The semantic type of the feature, indicating the kind of data it represents
            (e.g., "time", "cost", "categorical", etc.).

        optional: bool, optional
            Whether the feature is optional. If True, the feature can be left unset
            without causing errors. Default is False.

        default: _T or UNSET, optional
            The default value of the feature. If not provided, it is set to UNSET,
            indicating that the feature has no default data.

        owner: bool or None, optional
            Whether this feature owns its data.
            If True, the feature is responsible for providing its data and it
            will not be overwritten by user instance data.
            Features with owner=False are considered consumers, they expect their
            data to be provided by other features or the instance data.
            If None (default), ownership is determined based on whether a default
            value is provided.

        dynamic: bool, optional
            Whether the feature is dynamic, meaning that its data can change during the
            scheduling process. Default is False.

        shape: tuple[BaseShapeDim, ...] or None, optional
            The shape of the feature data, where BaseShapeDim can be an int or a
            symbolic dimension. If None, the shape is not specified. Default is None.

        n_categories: int or None, optional
            The number of categories for categorical features. Only applicable if
            semantic is "categorical". Default is None.

        low: float or None, optional
            The lower bound for numerical features, if applicable. Default is None.

        high: float or None, optional
            The upper bound for numerical features, if applicable. Default is None.


        Raises
        ------
        ValueError
            If ownership is explicitly set to False but a default value is provided.
            Consumers cannot own data.

        """
        owns_data = not is_unset(default)

        if owner is None:
            self.owner = owns_data

        elif not owner and owns_data:
            raise ValueError(
                f"Feature '{name}' is explicitly not an owner, but provides "
                "default data. A non-provider must have the default data unset."
            )

        else:
            self.owner = owner

        self.optional = optional
        self.name = name
        self.spec = FeatureSpec(
            scope=scope,
            semantic=semantic,
            shape=shape,
            n_categories=n_categories,
            low=low,
            high=high,
        )

        self._storage = default

        self.dynamic = dynamic
        self._data = default

    @classmethod
    def from_spec(
        cls,
        name: str,
        spec: FeatureSpec,
        *,
        optional: bool = False,
        default: _T | _UnsetType = UNSET,
        owner: bool | None = None,
        dynamic: bool = False,
    ) -> Self:
        """Create a feature instance from a given specification.

        Parameters
        ----------
        name: str
            The name of the feature.

        spec: FeatureSpec
            The specification for the feature.

        optional: bool, optional
            Whether the feature is optional. Default is False.

        default: _T or UNSET, optional
            The default value of the feature. Default is UNSET.

        owner: bool or None, optional
            Whether this feature owns its data. If None (default), ownership is determined
            based on whether a default value is provided.

        dynamic: bool, optional
            Whether the feature is dynamic. Default is False.

        Returns
        -------
        Feature[_T]
            A new feature instance based on the given specification.

        """
        return cls(
            name=name,
            scope=spec.scope,
            semantic=spec.semantic,
            optional=optional,
            default=default,
            owner=owner,
            dynamic=dynamic,
            shape=spec.raw_shape,
            n_categories=spec.n_categories,
            low=spec.low,
            high=spec.high,
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
    def shape(self) -> tuple[BaseShapeDim, ...] | None:
        """Get the feature's expected raw shape from its specification."""
        return self.spec.raw_shape

    def reset(self) -> None:
        """Reset the feature's data to its default value.

        When the feature is a consumer, resetting will clear any shared data.
        """
        self._data = self._storage

    def own_data(self, data: _T) -> None:
        """Set the feature's data as the owner of the data.

        This method turns the feature into an owner.
        If the feature is already an owner, it raises an error to prevent
        accidental overwriting of data.
        """
        if self.owner:
            raise ValueError(
                f"Feature '{self.name}' already is an owner, use `set_data` instead."
            )

        self.owner = True
        self._storage = data
        self._data = data

    def set_data(self, data: _T) -> None:
        """Set the feature's data."""
        if not self.dynamic and self.loaded:
            raise RuntimeError(
                f"Feature '{self.name}' already has loaded data: {self._data}."
            )

        self._data = data

    def shared_data(self, source: "Feature[_T]") -> None:
        """Get data from another feature, sharing the same data reference.

        The source feature must be compatible in terms of specification
        and must be an owner of its data.
        The current feature must not be an owner, as it will become a consumer
        sharing the source's data.
        """
        if not source.spec.shareable_with(self.spec):
            raise ValueError(
                f"Feature '{self.name}' has different specs from source: "
                f"expected {self.spec}, got {source.spec}."
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

    def validate(self, **symbol_values: int) -> None:
        """Validate the feature's loaded data against its specification."""
        if not self.loaded:
            if not self.optional:
                raise RuntimeError(
                    f"Feature {self.name} is required but has no loaded data."
                )

            return

        target_shape = self.spec.resolve_shape(**symbol_values)
        if target_shape is not None and not has_shape(self._data, target_shape):
            raise ValueError(
                f"Feature {self.name} has invalid shape: "
                f"expected {target_shape}, got {self._data}."
            )

    def __eq__(self, value: object, /) -> bool:
        """Check equality of features based on their name and specification."""
        return (
            isinstance(value, Feature)
            and self.name == value.name
            and self.spec == value.spec
        )


def _expand_shape(
    first_dim: BaseShapeDim, shape: tuple[BaseShapeDim, ...] | None
) -> tuple[BaseShapeDim, ...]:
    if shape is None:
        return (first_dim, None)

    return (first_dim, *shape)


class TaskFeature(Feature[list[_T]]):
    """Feature that applies to tasks in the scheduling instance.

    Task features are expected to have data that is a list of values, where each value
    corresponds to a specific task.
    The shape of the data is determined by the number of tasks in the instance,
    and any additional dimensions specified in the shape parameter.

    Example usage:
    >>> processing_times = TaskFeature(
    ...     name="processing_times",
    ...     semantic="time",
    ...     shape=(n_machines,),  # Shape will be (n_tasks, n_machines)
    ... )
    """

    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        *,
        optional: bool = False,
        default: list[_T] | _UnsetType = UNSET,
        owner: bool | None = None,
        dynamic: bool = False,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize a task feature with the given parameters.

        Parameters
        ----------
        name: str
            The name of the feature.

        semantic: str
            The semantic type of the feature, indicating the kind of data it represents
            (e.g., "time", "cost", "categorical", etc.).

        optional: bool, optional
            Whether the feature is optional. If True, the feature can be left unset
            without causing errors. Default is False.

        default: list[_T] or UNSET, optional
            The default value of the feature. If not provided, it is set to UNSET,
            indicating that the feature has no default data.

        owner: bool or None, optional
            Whether this feature owns its data.
            If True, the feature is responsible for providing its data and it
            will not be overwritten by user instance data.
            Features with owner=False are considered consumers, they expect their
            data to be provided by other features or the instance data.
            If None (default), ownership is determined based on whether a default
            value is provided.

        dynamic: bool, optional
            Whether the feature is dynamic, meaning that its data can change during the
            scheduling process. Default is False.

        shape: tuple[BaseShapeDim, ...] or None, optional
            The shape of the feature data, where BaseShapeDim can be an int or a
            symbolic dimension. The first dimension is reserved for the number of tasks
            and will be automatically prepended to the shape. If None, the shape is not
            specified.

        n_categories: int or None, optional
            The number of categories for categorical features. Only applicable if
            semantic is "categorical". Default is None.

        low: float or None, optional
            The lower bound for numerical features, if applicable. Default is None.

        high: float or None, optional
            The upper bound for numerical features, if applicable. Default is None.

        """
        scope: Literal["task"] = "task"

        super().__init__(
            name=name,
            scope=scope,
            semantic=semantic,
            optional=optional,
            default=default,
            owner=owner,
            dynamic=dynamic,
            shape=_expand_shape("n_tasks", shape),
            n_categories=n_categories,
            low=low,
            high=high,
        )


class JobFeature(Feature[list[_T]]):
    """Feature that applies to jobs in the scheduling instance.

    Job features are expected to have data that is a list of values, where each value
    corresponds to a specific job.
    The shape of the data is determined by the number of jobs in the instance,
    and any additional dimensions specified in the shape parameter.

    Example usage:
    >>> due_dates = JobFeature(
    ...     name="due_dates",
    ...     semantic="time",
    ...     shape=(),  # Shape will be (n_jobs,)
    ... )
    """

    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        optional: bool = False,
        owner: bool | None = None,
        default: list[_T] | _UnsetType = UNSET,
        dynamic: bool = False,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize a job feature with the given parameters.

        Parameters
        ----------
        name: str
            The name of the feature.

        semantic: str
            The semantic type of the feature, indicating the kind of data it represents
            (e.g., "time", "cost", "categorical", etc.).

        optional: bool, optional
            Whether the feature is optional. If True, the feature can be left unset
            without causing errors. Default is False.

        default: list[_T] or UNSET, optional
            The default value of the feature. If not provided, it is set to UNSET,
            indicating that the feature has no default data.

        owner: bool or None, optional
            Whether this feature owns its data.
            If True, the feature is responsible for providing its data and it
            will not be overwritten by user instance data.
            Features with owner=False are considered consumers, they expect their
            data to be provided by other features or the instance data.
            If None (default), ownership is determined based on whether a default
            value is provided.

        dynamic: bool, optional
            Whether the feature is dynamic, meaning that its data can change during the
            scheduling process. Default is False.

        shape: tuple[BaseShapeDim, ...] or None, optional
            The shape of the feature data, where BaseShapeDim can be an int or a
            symbolic dimension. The first dimension is reserved for the number of jobs
            and will be automatically prepended to the shape. If None, the shape is not
            specified.

        n_categories: int or None, optional
            The number of categories for categorical features. Only applicable if
            semantic is "categorical". Default is None.

        low: float or None, optional
            The lower bound for numerical features, if applicable. Default is None.

        high: float or None, optional
            The upper bound for numerical features, if applicable. Default is None.

        """
        scope: Literal["job"] = "job"

        super().__init__(
            name=name,
            scope=scope,
            semantic=semantic,
            optional=optional,
            owner=owner,
            default=default,
            dynamic=dynamic,
            shape=_expand_shape("n_jobs", shape),
            n_categories=n_categories,
            low=low,
            high=high,
        )


class MachineFeature(Feature[list[_T]]):
    """Feature that applies to machines in the scheduling instance.

    Machine features are expected to have data that is a list of values, where each value
    corresponds to a specific machine.
    The shape of the data is determined by the number of machines in the instance,
    and any additional dimensions specified in the shape parameter.

    Example usage:
    >>> machine_speeds = MachineFeature(
    ...     name="machine_speeds",
    ...     semantic="discrete",
    ...     shape=(),  # Shape will be (n_machines,)
    ... )
    """

    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        optional: bool = False,
        owner: bool | None = None,
        default: list[_T] | _UnsetType = UNSET,
        dynamic: bool = False,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize a machine feature with the given parameters.

        Parameters
        ----------
        name: str
            The name of the feature.

        semantic: str
            The semantic type of the feature, indicating the kind of data it represents
            (e.g., "time", "cost", "categorical", etc.).

        optional: bool, optional
            Whether the feature is optional. If True, the feature can be left unset
            without causing errors. Default is False.

        default: list[_T] or UNSET, optional
            The default value of the feature. If not provided, it is set to UNSET,
            indicating that the feature has no default data.

        owner: bool or None, optional
            Whether this feature owns its data.
            If True, the feature is responsible for providing its data and it
            will not be overwritten by user instance data.
            Features with owner=False are considered consumers, they expect their
            data to be provided by other features or the instance data.
            If None (default), ownership is determined based on whether a default
            value is provided.

        dynamic: bool, optional
            Whether the feature is dynamic, meaning that its data can change during the
            scheduling process. Default is False.

        shape: tuple[BaseShapeDim, ...] or None, optional
            The shape of the feature data, where BaseShapeDim can be an int or a
            symbolic dimension. The first dimension is reserved for the number of machines
            and will be automatically prepended to the shape. If None, the shape is not
            specified.

        n_categories: int or None, optional
            The number of categories for categorical features. Only applicable if
            semantic is "categorical". Default is None.

        low: float or None, optional
            The lower bound for numerical features, if applicable. Default is None.

        high: float or None, optional
            The upper bound for numerical features, if applicable. Default is None.

        """
        scope: Literal["machine"] = "machine"

        super().__init__(
            name=name,
            scope=scope,
            semantic=semantic,
            optional=optional,
            owner=owner,
            default=default,
            dynamic=dynamic,
            shape=_expand_shape("n_machines", shape),
            n_categories=n_categories,
            low=low,
            high=high,
        )


class GlobalFeature(Feature[_T]):
    """Feature that applies to the entire scheduling instance.

    Global features can be any type of data that is relevant to the scheduling
    instance as a whole, such as time, graph structure, or other global parameters.

    Example usage:
    >>> time_horizon = GlobalFeature(
    ...     name="time_horizon",
    ...     semantic="time",
    ...     shape=(),  # Shape will be ()
    ... )
    """

    def __init__(
        self,
        name: str,
        semantic: SemanticType,
        optional: bool = False,
        owner: bool | None = None,
        default: _T | _UnsetType = UNSET,
        dynamic: bool = False,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        """Initialize a global feature with the given parameters.

        Parameters
        ----------
        name: str
            The name of the feature.

        semantic: str
            The semantic type of the feature, indicating the kind of data it represents
            (e.g., "time", "cost", "categorical", etc.).

        optional: bool, optional
            Whether the feature is optional. If True, the feature can be left unset
            without causing errors. Default is False.

        default: list[_T] or UNSET, optional
            The default value of the feature. If not provided, it is set to UNSET,
            indicating that the feature has no default data.

        owner: bool or None, optional
            Whether this feature owns its data.
            If True, the feature is responsible for providing its data and it
            will not be overwritten by user instance data.
            Features with owner=False are considered consumers, they expect their
            data to be provided by other features or the instance data.
            If None (default), ownership is determined based on whether a default
            value is provided.

        dynamic: bool, optional
            Whether the feature is dynamic, meaning that its data can change during the
            scheduling process. Default is False.

        shape: tuple[BaseShapeDim, ...] or None, optional
            The shape of the feature data, where BaseShapeDim can be an int or a
            symbolic dimension. If None, the shape is not specified.

        n_categories: int or None, optional
            The number of categories for categorical features. Only applicable if
            semantic is "categorical". Default is None.

        low: float or None, optional
            The lower bound for numerical features, if applicable. Default is None.

        high: float or None, optional
            The upper bound for numerical features, if applicable. Default is None.

        """
        scope: Literal["global"] = "global"

        super().__init__(
            name=name,
            scope=scope,
            semantic=semantic,
            optional=optional,
            owner=owner,
            default=default,
            dynamic=dynamic,
            shape=shape,
            n_categories=n_categories,
            low=low,
            high=high,
        )

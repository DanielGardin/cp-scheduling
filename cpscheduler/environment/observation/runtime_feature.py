"""Feature classes for scheduling instance specifications and data management."""

from typing import Any, Generic

from typing_extensions import TypeVar

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.specs.feature_spec import (
    FeatureSpec,
    Scope,
    SemanticType,
)
from cpscheduler.environment.specs.symbols import BaseShapeDim

_T = TypeVar("_T", default=Any)


class RuntimeFeature(EzPickle, Generic[_T]):
    """Feature that is derived from the current state of the environment.

    Differently from features defined by the Feature class, runtime features
    are not loaded from the instance data, or from a component, but rather
    computed in observation time, based on the current state.

    They are inherently dynamic, and its life cycle is tied to the observation
    that defines them.
    """

    name: str
    spec: FeatureSpec
    value: _T

    def __init__(
        self,
        name: str,
        scope: Scope,
        semantic: SemanticType,
        data: _T,
        *,
        shape: tuple[BaseShapeDim, ...] | None = None,
        n_categories: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ):
        """Initialize the RuntimeFeature.

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

        data: _T
            The initial value of the feature.

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

        """
        self.name = name
        self.spec = FeatureSpec(
            scope=scope,
            semantic=semantic,
            shape=shape,
            n_categories=n_categories,
            low=low,
            high=high,
        )
        self.value = data

"""Base specification for observation structures."""

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class ObservationSpec(EzPickle):
    """Base specification node for observation structures."""

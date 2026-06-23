"""Tracer module for the CPScheduler environment.

Tracers are used to monitor internal state before each decision step, logging
information about the state of the environment when an action is taken.

We make available a few tracers for common use cases, but users can implement
their own tracers by subclassing the `Tracer` class.
"""

__all__ = [
    "ExecutionTrajectoryTracer",
    "FullTrajectoryTracer",
    "Tracer",
]

from .base import Tracer
from .trajectory import ExecutionTrajectoryTracer, FullTrajectoryTracer

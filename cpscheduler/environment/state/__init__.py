"""State components for the scheduling environment.

This submodule exposes the core data structures used to represent the
constraint-propagation and discrete-event simulation state of a scheduling
problem. The primary entrypoint is `ScheduleState`, which bundles the
following concerns:

- Instance view: read-only problem parameters (from :class:`ProblemInstance`).
- CSP domains: per-(task,machine) variable domains and presence flags.
- DES runtime: execution history, task statuses and runtime event queues.

The public API is intentionally small: constraints, objectives and setups use
the methods exposed by ``ScheduleState`` and the event containers rather than
directly manipulating internal structures.
"""

__all__ = ["ScheduleState"]

from .state import ScheduleState

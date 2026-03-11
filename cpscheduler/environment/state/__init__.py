"""
This module contains all the components related to the state of the scheduling
environment.

The full state is represented by the ScheduleState class, which aggregates
instance information, variable assignments, and the current runtime state of the
tasks.

This state only exposes the necessary API for constraints, objectives and setups
to interact with, enabling the environment to orchestrate the simulation and
accept changes from the scheduler without exposing the internal details of the
state representation.
"""

__all__ = ["ScheduleState", "ObsType"]

from .state import ScheduleState, ObsType

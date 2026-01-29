from enum import Enum, auto
from dataclasses import dataclass

from cpscheduler.environment._common import TIME, MACHINE_ID
from cpscheduler.environment.tasks import Task


class VarField(Enum):
    START_LB = auto()
    START_UB = auto()
    END_LB = auto()
    END_UB = auto()


class InfeasibleDecision(Exception):
    """
    Exception raised when an infeasible decision is made in the scheduling environment.
    """

    pass


@dataclass
class Event:
    """
    Base class for events in the scheduling environment.
    """

    task: Task
    field: VarField
    value: TIME
    machine_id: MACHINE_ID = -1

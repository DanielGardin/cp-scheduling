from typing import Any, Optional, Literal, overload, Callable, Sequence
from numpy.typing import NDArray
from pandas import DataFrame

import numpy as np

from copy import deepcopy

from .environment import SchedulingCPEnv, PrecedenceConstraint, NonOverlapConstraint, Makespan
from .environment.vector import SyncVectorEnv, AsyncVectorEnv

from .environment.constraints import Constraint
from .environment.objectives import Objective
from .environment.variables import IntervalVars

known_envs: dict[str, type[SchedulingCPEnv]] = {}


def register_env(env: type[SchedulingCPEnv], name: Optional[str] = None) -> None:
    if name is None:
        name = env.__name__

    known_envs[name] = env

class JobShopEnv(SchedulingCPEnv):
    def __init__(
            self,
            instance: DataFrame,
            duration: str | NDArray[np.int32] = 'processing_time',
            job_feature: str = 'job',
            operation_feature: str = 'operation',
            machine_feature: str = 'machine'
        ) -> None:
        super().__init__(instance, duration)
        self.add_constraint(
            PrecedenceConstraint.jobshop_precedence(self.tasks, job_feature, operation_feature)
        )

        self.add_constraint(
            NonOverlapConstraint.jobshop_non_overlap(self.tasks, machine_feature)
        )

        self.set_objective(
            Makespan(self.tasks)
        )

    def render(self) -> None:
        return self.render_gantt(
            'machine',
            'job',
        )

register_env(JobShopEnv, 'jobshop')

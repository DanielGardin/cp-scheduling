from typing import Any, Optional, Literal, overload, Callable, Sequence
from numpy.typing import NDArray, ArrayLike
from pandas import DataFrame

import numpy as np

from copy import deepcopy


from .environment.constraints import PrecedenceConstraint, NonOverlapConstraint, ResourceCapacityConstraint
from .environment.objectives import Makespan
from .environment import register_env, SchedulingCPEnv


class JobShopEnv(SchedulingCPEnv):
    def __init__(
            self,
            instance: DataFrame,
            duration: str | ArrayLike = 'processing_time',
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
            Makespan
        )

    def render(self) -> None:
        return self.render_gantt(
            'machine',
            'job',
        )

class ResourceConstraintEnv(SchedulingCPEnv):
    def __init__(
            self,
            instance: DataFrame,
            capacity: ArrayLike,
            precedence: Sequence[Sequence[int]],
            duration: str | NDArray[np.int32] = 'processing_time',
            resource_features: str | list[str] = 'resource',

        ) -> None:
        super().__init__(instance, duration)

        self.add_constraint(
            PrecedenceConstraint,
            precedence_list=precedence
        )

        self.add_constraint(
            ResourceCapacityConstraint,
        )

        self.set_objective(
            Makespan
        )

    def render(self) -> None:
        return self.render_gantt()
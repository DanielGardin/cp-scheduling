from typing import Sequence, Iterable
from pandas import DataFrame

from .environment.constraints import PrecedenceConstraint, NonOverlapConstraint, ResourceCapacityConstraint
from .environment.objectives import Makespan
from .environment import SchedulingCPEnv


class JobShopEnv(SchedulingCPEnv):
    def __init__(
            self,
            instance: DataFrame,
            duration: str | Iterable[int] = 'processing_time',
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
            capacity: Iterable[float],
            precedence: Sequence[Sequence[int]],
            duration: str | Iterable[int] = 'processing_time',
            resource_features: str | list[str] = 'resource',

        ) -> None:
        super().__init__(instance, duration)

        self.add_constraint(
            PrecedenceConstraint,
            precedence_list=precedence
        )

        self.add_constraint(
            ResourceCapacityConstraint,
            capacity=capacity,
        )

        self.set_objective(
            Makespan
        )

    def render(self) -> None:
        return self.render_gantt()
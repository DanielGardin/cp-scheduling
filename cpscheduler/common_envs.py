from typing import Any, Sequence, Iterable
from pandas import DataFrame

from .environment.constraints import PrecedenceConstraint, NonOverlapConstraint, ResourceCapacityConstraint
from .environment.objectives import Makespan, WeightedCompletionTime
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
            PrecedenceConstraint.jobshop_precedence,
            job_feature,
            operation_feature
        )

        self.add_constraint(
            NonOverlapConstraint.jobshop_non_overlap,
            machine_feature
        )

        self.set_objective(
            Makespan
        )


    def _get_info(self) -> dict[str, Any]:
        original_info = super()._get_info()

        total_executed_time = sum(self.tasks.get_duration(self.get_fixed_tasks()))

        return {
            **original_info,
            'speedup':  total_executed_time / self.current_time if self.current_time > 0 else 0
        }


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
            resource_features: str | Iterable[str] | Iterable[float] | Iterable[Iterable[float]] = 'resource',

        ) -> None:
        super().__init__(instance, duration)

        self.add_constraint(
            PrecedenceConstraint,
            precedence_list=precedence
        )

        self.add_constraint(
            ResourceCapacityConstraint,
            resources=resource_features,
            resource_capacity=capacity,
        )

        self.set_objective(
            Makespan
        )

    def render(self) -> None:
        return self.render_gantt()


class CustomerJobShopEnv(SchedulingCPEnv):
    def __init__(
            self,
            instance: DataFrame,
            customer_weights: Iterable[float],
            duration: str | Iterable[int] = 'processing_time',
            job_feature: str = 'job',
            operation_feature: str = 'operation',
            machine_feature: str = 'machine',
            customer_feature: str = 'customer',
        ):
        super().__init__(instance, duration)
        self.add_constraint(
            PrecedenceConstraint.jobshop_precedence,
            job_feature,
            operation_feature
        )

        self.add_constraint(
            NonOverlapConstraint.jobshop_non_overlap,
            machine_feature
        )

        self.set_objective(
            WeightedCompletionTime
        )
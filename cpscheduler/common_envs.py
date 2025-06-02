from pandas import DataFrame

from .environment import SchedulingCPEnv, PrecedenceConstraint, NonOverlapConstraint, Makespan

class JobShopEnv(SchedulingCPEnv):
    def __init__(
            self,
            instance: DataFrame,
            duration_feature: str | int = 'processing_time',
            job_feature: str = 'job',
            operation_feature: str = 'operation',
            machine_feature: str = 'machine',
            dataframe_obs: bool = True
        ) -> None:
        super().__init__(instance, duration_feature, dataframe_obs)
        self.add_constraint(
            PrecedenceConstraint.jobshop_precedence(self.tasks, job_feature, operation_feature)
        )
        self.add_constraint(
            NonOverlapConstraint.jobshop_non_overlap(self.tasks, machine_feature)
        )
        self.add_objective(
            Makespan(self.tasks)
        )
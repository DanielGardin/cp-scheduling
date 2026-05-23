from cpscheduler.environment.constants import MachineID, Time
from cpscheduler.environment.constraints import (
    Constraint,
    MachineConstraint,
    NonOverlapConstraint,
    PrecedenceConstraint,
)
from cpscheduler.environment.instance import ProblemInstance, TaskFeature
from cpscheduler.environment.setups.base import ScheduleSetup
from cpscheduler.environment.setups.parallel import (
    UnrelatedParallelMachineSetup,
)


class OpenShopSetup(ScheduleSetup):
    """
    Open Shop Scheduling Setup.

    This setup is used for scheduling tasks in an open shop environment where each task can be
    processed on any machine, and the order of operations is not fixed.
    """

    processing_times: TaskFeature[Time]
    machines: TaskFeature[MachineID]
    disjunctive: bool

    def __init__(
        self,
        processing_times: str = "processing_time",
        machine_feature: str = "machine",
        disjunctive: bool = True,
    ):
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times,
            elem_type=Time,
            semantic="duration",
        )

        self.machines = TaskFeature(
            name=machine_feature,
            elem_type=MachineID,
            semantic="machine",
        )

    @property
    def n_machines(self) -> int:
        if self.machines.loaded:
            return max(self.machines.value) + 1

        return 0

    def get_features(self) -> list[TaskFeature]:
        return [self.processing_times, self.machines]

    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, (p_time, machine_id) in enumerate(
            zip(self.processing_times.value, self.machines.value, strict=False)
        ):
            instance.set_processing_time(task_id, machine_id, p_time)

    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        task_disjunction = NonOverlapConstraint(task_groups=instance.job_tasks)

        return (
            (MachineConstraint(), task_disjunction)
            if self.disjunctive
            else (task_disjunction,)
        )

    def get_entry(self) -> str:
        if self.machines.loaded:
            return f"O{self.n_machines}"

        return "Om"

    @classmethod
    def get_general_entry(cls) -> str:
        return "Om"


def build_job_precedence(
    instance: ProblemInstance, operation_order: list[int], name: str
) -> PrecedenceConstraint:
    precedence = PrecedenceConstraint(name=name)

    task_orders = [[-1] * len(tasks) for tasks in instance.job_tasks]

    for task_id, (job, op) in enumerate(
        zip(instance.job_ids, operation_order, strict=False)
    ):
        if task_orders[job][op] != -1:
            raise ValueError(
                f"Cannot have tasks have the same job and operation values: "
                f" {job}, {op}"
            )

        task_orders[job][op] = task_id

    for chain in task_orders:
        precedence.add_chain(chain)

    return precedence


class JobShopSetup(OpenShopSetup):
    """
    Job Shop Scheduling Setup.

    This setup is used for scheduling tasks in a job shop environment where each task has a specific
    operation order and is assigned to a specific machine.
    """

    operation_order: TaskFeature[int]

    def __init__(
        self,
        processing_times: str = "processing_time",
        operation_order: str = "operation",
        machine_feature: str = "machine",
        disjunctive: bool = True,
    ):
        super().__init__(
            processing_times=processing_times,
            machine_feature=machine_feature,
            disjunctive=disjunctive,
        )

        self.operation_order = TaskFeature(
            name=operation_order,
            elem_type=int,
            semantic="order",
        )

    def get_features(self) -> list[TaskFeature]:
        return [
            self.processing_times,
            self.machines,
            self.operation_order,
        ]

    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        precedence = build_job_precedence(
            instance, self.operation_order.value, "jobshop_chains"
        )

        return (
            (MachineConstraint(), precedence)
            if self.disjunctive
            else (precedence,)
        )

    def get_entry(self) -> str:
        if self.machines.loaded:
            return f"J{self.n_machines}"

        return "Jm"

    @classmethod
    def get_general_entry(cls) -> str:
        return "Jm"


class FlexibleJobShopSetup(UnrelatedParallelMachineSetup):
    """
    Flexible Job Shop Scheduling Setup.

    This setup is a variant of the job shop scheduling problem, where each task
    can be processed on one of several machines instead of a specific machine.
    The operation order is still fixed, but the machine assignment is flexible,
    allowing for more scheduling options and potentially better solutions.
    """

    operation_order: TaskFeature[int]

    def __init__(
        self,
        processing_times: str = "processing_times",
        operation_order: str = "operation",
        disjunctive: bool = True,
    ):
        super().__init__(
            processing_times=processing_times,
            disjunctive=disjunctive,
        )

        self.operation_order = TaskFeature(
            name=operation_order,
            elem_type=int,
            semantic="order",
        )

    def get_features(self) -> list[TaskFeature]:
        return [
            self.processing_times,
            self.operation_order,
        ]

    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        precedence = build_job_precedence(
            instance, self.operation_order.value, "flexible_jobshop_chains"
        )

        return (
            (MachineConstraint(), precedence)
            if self.disjunctive
            else (precedence,)
        )

    def get_entry(self) -> str:
        if self.processing_times.loaded:
            return f"FJ{self.n_machines}"

        return "FJm"

    @classmethod
    def get_general_entry(cls) -> str:
        return "FJm"


class FlowShopSetup(ScheduleSetup):
    """
    Flow Shop Scheduling Setup.

    This setup is used for scheduling tasks in a flow shop environment where
    each task has a specific operation order and all tasks follow the same
    machine order.
    """

    processing_times: TaskFeature[Time]
    operation_order: TaskFeature[int]
    disjunctive: bool

    def __init__(
        self,
        processing_times: str = "processing_time",
        operation_order: str = "operation",
        disjunctive: bool = True,
    ):
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times,
            elem_type=Time,
            semantic="duration",
        )

        self.operation_order = TaskFeature(
            name=operation_order,
            elem_type=int,
            semantic="order",
        )

    @property
    def n_machines(self) -> int:
        if self.operation_order.loaded:
            return max(self.operation_order.value) + 1

        return 0

    def get_features(self) -> list[TaskFeature]:
        return [
            self.processing_times,
            self.operation_order,
        ]

    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        precedence = build_job_precedence(
            instance, self.operation_order.value, "flowshop_chains"
        )

        return (
            (MachineConstraint(), precedence)
            if self.disjunctive
            else (precedence,)
        )

    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, (p_time, machine_id) in enumerate(
            zip(
                self.processing_times.value,
                self.operation_order.value,
                strict=False,
            )
        ):
            instance.set_processing_time(task_id, machine_id, p_time)

    def get_entry(self) -> str:
        if self.operation_order.loaded:
            return f"F{self.n_machines}"

        return "Fm"

    @classmethod
    def get_general_entry(cls) -> str:
        return "Fm"

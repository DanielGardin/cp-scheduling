"""Scheduling problem setups for shop scheduling problems.

Job shop problems are a class of scheduling problems where a set of jobs,
each consisting of a sequence of tasks, must be scheduled on a set of machines.

Each task requires processing on a specific machine for a certain duration,
and the goal is to optimize a scheduling objective by selecting the start times
of each task while respecting the constraints of the problem.
"""

from typing import override

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
    """Open Shop Scheduling Setup.

    This setup is used for scheduling tasks in an open shop environment where
    each task can be processed on any machine, and the order of operations is not fixed.
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
        """Initialize the Open Shop Scheduling Setup.

        Parameters
        ----------
        processing_times: str
            The name of the task feature that contains the processing times.

        machine_feature: str
            The name of the task feature that contains the machine assignments for each task.

        disjunctive: bool
            Whether to include disjunctive constraints for the machines.
            If False, machines can process multiple tasks simultaneously.
            When disjunctive is True, the setup will include constraints to ensure
            that only one task can be processed at a time on each machine.

        """
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times,
            semantic="duration",
            shape=(),
        )

        self.machines = TaskFeature(
            name=machine_feature,
            semantic="machine",
            shape=(),
        )

    @property
    @override
    def n_machines(self) -> int:
        if self.machines.loaded:
            return max(self.machines.value) + 1

        return 0

    @override
    def get_features(self) -> list[TaskFeature]:
        return [self.processing_times, self.machines]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, (p_time, machine_id) in enumerate(
            zip(self.processing_times.value, self.machines.value, strict=False)
        ):
            instance.set_processing_time(task_id, machine_id, p_time)

    @override
    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        task_disjunction = NonOverlapConstraint(task_groups=instance.job_tasks)

        return (
            (MachineConstraint(), task_disjunction)
            if self.disjunctive
            else (task_disjunction,)
        )

    @override
    def get_entry(self) -> str:
        if self.machines.loaded:
            return f"O{self.n_machines}"

        return "Om"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Om"


def _build_job_precedence(
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
    """Job Shop Scheduling Setup.

    This setup is used for scheduling tasks in a job shop environment where each
    task has a specific operation order and is assigned to a specific machine.
    """

    operation_order: TaskFeature[int]

    def __init__(
        self,
        processing_times: str = "processing_time",
        operation_order: str = "operation",
        machine_feature: str = "machine",
        disjunctive: bool = True,
    ):
        """Initialize the Job Shop Scheduling Setup.

        Parameters
        ----------
        processing_times: str
            The name of the task feature that contains the processing times.

        operation_order: str
            The name of the task feature that contains the operation order for each task.

        machine_feature: str
            The name of the task feature that contains the machine assignments for each task.

        disjunctive: bool
            Whether to include disjunctive constraints for the machines.
            If False, machines can process multiple tasks simultaneously.
            When disjunctive is True, the setup will include constraints to ensure
            that only one task can be processed at a time on each machine.

        """
        super().__init__(
            processing_times=processing_times,
            machine_feature=machine_feature,
            disjunctive=disjunctive,
        )

        self.operation_order = TaskFeature(
            name=operation_order, semantic="order", shape=()
        )

    @override
    def get_features(self) -> list[TaskFeature]:
        return [
            self.processing_times,
            self.machines,
            self.operation_order,
        ]

    @override
    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        precedence = _build_job_precedence(
            instance, self.operation_order.value, "jobshop_chains"
        )

        return (
            (MachineConstraint(), precedence)
            if self.disjunctive
            else (precedence,)
        )

    @override
    def get_entry(self) -> str:
        if self.machines.loaded:
            return f"J{self.n_machines}"

        return "Jm"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Jm"


class FlexibleJobShopSetup(UnrelatedParallelMachineSetup):
    """Flexible Job Shop Scheduling Setup.

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
        """Initialize the Flexible Job Shop Scheduling Setup.

        Parameters
        ----------
        processing_times: str
            The name of the task feature that contains the processing times.

        operation_order: str
            The name of the task feature that contains the operation order for each task.

        disjunctive: bool
            Whether to include disjunctive constraints for the machines.
            If False, machines can process multiple tasks simultaneously.
            When disjunctive is True, the setup will include constraints to ensure
            that only one task can be processed at a time on each machine.

        """
        super().__init__(
            processing_times=processing_times,
            disjunctive=disjunctive,
        )

        self.operation_order = TaskFeature(
            name=operation_order, semantic="order", shape=()
        )

    @override
    def get_features(self) -> list[TaskFeature]:
        return [
            self.processing_times,
            self.operation_order,
        ]

    @override
    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        precedence = _build_job_precedence(
            instance, self.operation_order.value, "flexible_jobshop_chains"
        )

        return (
            (MachineConstraint(), precedence)
            if self.disjunctive
            else (precedence,)
        )

    @override
    def get_entry(self) -> str:
        if self.processing_times.loaded:
            return f"FJ{self.n_machines}"

        return "FJm"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "FJm"


class FlowShopSetup(ScheduleSetup):
    """Flow Shop Scheduling Setup.

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
        """Initialize the Flow Shop Scheduling Setup.

        Parameters
        ----------
        processing_times: str
            The name of the task feature that contains the processing times.

        operation_order: str
            The name of the task feature that contains the operation order for each task.

        disjunctive: bool
            Whether to include disjunctive constraints for the machines.
            If False, machines can process multiple tasks simultaneously.
            When disjunctive is True, the setup will include constraints to ensure
            that only one task can be processed at a time on each machine.

        """
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times, semantic="duration", shape=()
        )

        self.operation_order = TaskFeature(
            name=operation_order, semantic="order", shape=()
        )

    @property
    @override
    def n_machines(self) -> int:
        if self.operation_order.loaded:
            return max(self.operation_order.value) + 1

        return 0

    @override
    def get_features(self) -> list[TaskFeature]:
        return [
            self.processing_times,
            self.operation_order,
        ]

    @override
    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        precedence = _build_job_precedence(
            instance, self.operation_order.value, "flowshop_chains"
        )

        return (
            (MachineConstraint(), precedence)
            if self.disjunctive
            else (precedence,)
        )

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, (p_time, machine_id) in enumerate(
            zip(
                self.processing_times.value,
                self.operation_order.value,
                strict=False,
            )
        ):
            instance.set_processing_time(task_id, machine_id, p_time)

    @override
    def get_entry(self) -> str:
        if self.operation_order.loaded:
            return f"F{self.n_machines}"

        return "Fm"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Fm"

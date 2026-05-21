"""
schedule_setup.py

This module defines the ScheduleSetup class and its subclasses for different scheduling setups.
It provides a framework for creating various scheduling environments, such as single machine,
identical parallel machines, uniform parallel machines, job shop, and open shop setups.
"""

from collections.abc import Iterable

from mypy_extensions import mypyc_attr

from cpscheduler.environment.utils.general import convert_to_list

from cpscheduler.environment.constants import MachineID, Time, Int
from cpscheduler.environment.component import Component

from cpscheduler.environment.instance import (
    ProblemInstance, Feature, TaskFeature, MachineFeature, UNSET
)
from cpscheduler.environment.constraints import (
    Constraint, NonOverlapConstraint, PrecedenceConstraint, MachineConstraint
)

setups: dict[str, type["ScheduleSetup"]] = {}


def ceil_div(a: Time, b: Time) -> Time:
    "a divided by b, rounded up to the nearest integer."
    return -(-a // b)


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class ScheduleSetup(Component):
    """
    Base class for scheduling setups. It defines the common interface for all scheduling setups
    and provides methods to parse process times, set tasks, and setup constraints.
    """

    def __init_subclass__(cls) -> None:
        name = cls.__name__

        if not name.startswith('_'):
            setups[name] = cls

    @property
    def n_machines(self) -> int:
        "Return the number of machines after the instance is loaded."
        return 0

    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        "Build the constraints for that setup."
        return ()


class SingleMachineSetup(ScheduleSetup):
    """
    Single Machine Scheduling Setup.

    This setup is used for scheduling tasks on a single machine.
    """

    processing_times: TaskFeature[Time]

    disjunctive: bool

    def __init__(
        self,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ) -> None:
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times,
            elem_type=Time,
            semantic="duration",
        )

    @property
    def n_machines(self) -> int:
        return 1

    def get_features(self) -> list[TaskFeature]:
        return [self.processing_times]

    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, p_time in enumerate(self.processing_times.value):
            instance.set_processing_time(task_id, 0, p_time)

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        if not self.disjunctive:
            return ()

        return (MachineConstraint(),)

    @classmethod
    def get_general_entry(cls) -> str:
        return "1"


class IdenticalParallelMachineSetup(ScheduleSetup):
    """
    Identical Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines using the same processing time.
    """

    __args__ = ("n_machines",)

    _n_machines: int
    processing_times: TaskFeature[Time]
    disjunctive: bool

    def __init__(
        self,
        n_machines: int,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times,
            elem_type=Time,
            semantic="duration",
        )

        self._n_machines = n_machines

    @property
    def n_machines(self) -> int:
        return self._n_machines

    def get_features(self) -> list[TaskFeature]:
        return [self.processing_times]

    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, p_time in enumerate(self.processing_times.value):
            for machine in range(self.n_machines):
                instance.set_processing_time(task_id, machine, p_time)

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    def get_entry(self) -> str:
        return f"P{self.n_machines}"

    @classmethod
    def get_general_entry(cls) -> str:
        return "Pm"


class UniformParallelMachineSetup(ScheduleSetup):
    """
    Uniform Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines with different speeds.
    """

    __args__ = ("speed",)

    speed: MachineFeature[int]
    processing_times: TaskFeature[Time]
    disjunctive: bool

    def __init__(
        self,
        speed: Iterable[Int] | None = None,
        speed_tag: str = "speed",
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times,
            elem_type=Time,
            semantic="duration",
        )

        self.speed = MachineFeature(
            name=speed_tag,
            elem_type=int,
            semantic="discrete",
            default=(
                convert_to_list(speed, int)
                if speed is not None else UNSET
            )
        )

    @property
    def n_machines(self) -> int:
        if self.speed.loaded:
            return len(self.speed.value)

        return 0

    def get_features(self) -> list[Feature]:
        return [
            self.speed,
            self.processing_times,
        ]

    def initialize(self, instance: ProblemInstance) -> None:
        if any(s <= 0 for s in self.speed.value):
            raise ValueError("Machine speeds must be positive integers.")

        for task_id, p_time in enumerate(self.processing_times.value):
            for machine, speed in enumerate(self.speed.value):
                machine_p_time = ceil_div(p_time, speed)

                instance.set_processing_time(task_id, machine, machine_p_time)

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    def get_entry(self) -> str:
        if self.speed.loaded:
            return f"Q{self.n_machines}"

        return f"Qm"

    @classmethod
    def get_general_entry(cls) -> str:
        return "Qm"


class UnrelatedParallelMachineSetup(ScheduleSetup):

    __args__ = ("processing_times",)

    processing_times: TaskFeature[list[Time]]
    disjunctive: bool

    def __init__(
        self,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name="processing_times",
            elem_type=list[Time],
            semantic="duration",
        )

    @property
    def n_machines(self) -> int:
        if self.processing_times.loaded:
            return len(self.processing_times.value[0])
        
        return 0

    def get_features(self) -> list[TaskFeature]:
        return [self.processing_times]

    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, machine_times in enumerate(self.processing_times.value):
            for machine_id, ptime in enumerate(machine_times):
                instance.set_processing_time(
                    task_id,
                    machine_id,
                    ptime
                )

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    def get_entry(self) -> str:
        if self.processing_times.loaded:
            return f"R{self.n_machines}"

        return "Rm"

    @classmethod
    def get_general_entry(cls) -> str:
        return "Rm"


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
            zip(self.processing_times.value, self.machines.value)
        ):
            instance.set_processing_time(task_id, machine_id, p_time)

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
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
    instance: ProblemInstance,
    operation_order: list[int],
    name: str
) -> PrecedenceConstraint:
    precedence = PrecedenceConstraint(name=name)

    task_orders = [
        [-1] * len(tasks) for tasks in instance.job_tasks
    ]

    for task_id, (job, op) in enumerate(
        zip(instance.job_ids.value, operation_order)
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
        self,
        instance: ProblemInstance
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
        self,
        instance: ProblemInstance
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
        self,
        instance: ProblemInstance
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
            zip(self.processing_times.value, self.operation_order.value)
        ):
            instance.set_processing_time(task_id, machine_id, p_time)


    def get_entry(self) -> str:
        if self.operation_order.loaded:
            return f"F{self.n_machines}"

        return "Fm"

    @classmethod
    def get_general_entry(cls) -> str:
        return "Fm"

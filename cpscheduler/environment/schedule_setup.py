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
from cpscheduler.environment.components import Component

from cpscheduler.environment.instance import ProblemInstance
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

    processing_times: str
    disjunctive: bool

    def __init__(
        self,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ) -> None:
        self.processing_times = processing_times
        self.disjunctive = disjunctive

    def initialize(self, instance: ProblemInstance) -> None:
        p_times = instance.register_task_feature(self.processing_times)

        for task_id, p_time in enumerate(p_times):
            instance.set_processing_time(task_id, 0, Time(p_time))

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        if not self.disjunctive:
            return ()

        return (MachineConstraint(),)

    def get_entry(self) -> str:
        return "1"


class IdenticalParallelMachineSetup(ScheduleSetup):
    """
    Identical Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines using the same processing time.
    """

    __args__ = ("n_machines",)

    n_machines: int
    processing_times: str 
    disjunctive: bool

    def __init__(
        self,
        n_machines: int,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.n_machines = n_machines
        self.processing_times = processing_times
        self.disjunctive = disjunctive

    def initialize(self, instance: ProblemInstance) -> None:
        p_times = instance.register_task_feature(self.processing_times)

        for task_id, p_time in enumerate(p_times):
            for machine in range(self.n_machines):
                instance.set_processing_time(task_id, machine, Time(p_time))

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    def get_entry(self) -> str:
        return "Pm"


class UniformParallelMachineSetup(ScheduleSetup):
    """
    Uniform Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines with different speeds.
    """

    __args__ = ("speed",)

    speed: list[int]
    processing_times: str
    disjunctive: bool

    def __init__(
        self,
        speed: Iterable[Int],
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.speed = convert_to_list(speed, int)

        if any(s <= 0 for s in self.speed):
            raise ValueError("Machine speeds must be positive integers.")

        self.processing_times = processing_times
        self.disjunctive = disjunctive

    def initialize(self, instance: ProblemInstance) -> None:
        instance.register_global_feature("speed", self.speed)
        p_times = instance.register_task_feature(self.processing_times)

        for task_id, p_time in enumerate(p_times):
            for machine, speed in enumerate(self.speed):
                machine_p_time = ceil_div(Time(p_time), speed)

                instance.set_processing_time(task_id, machine, machine_p_time)

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    def get_entry(self) -> str:
        return "Qm"


class UnrelatedParallelMachineSetup(ScheduleSetup):

    __args__ = ("processing_times",)

    processing_times: list[str]
    disjunctive: bool

    def __init__(
        self,
        processing_times: Iterable[str],
        disjunctive: bool = True,
    ):
        if isinstance(processing_times, str):
            raise ValueError(
                "UnrelatedParallelMachineSetup does not support a single processing time feature. "
                "Please provide an iterable of processing time features."
            )

        self.processing_times = list(processing_times)
        self.disjunctive = disjunctive

    def initialize(self, instance: ProblemInstance) -> None:
        for machine, p_time_feature in enumerate(self.processing_times):
            p_times = instance.register_task_feature(p_time_feature)

            for task_id, p_time in enumerate(p_times):
                instance.set_processing_time(task_id, machine, Time(p_time))

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    def get_entry(self) -> str:
        return "Rm"


class OpenShopSetup(ScheduleSetup):
    """
    Open Shop Scheduling Setup.

    This setup is used for scheduling tasks in an open shop environment where each task can be
    processed on any machine, and the order of operations is not fixed.
    """

    processing_times: str
    machine_feature: str
    disjunctive: bool

    def __init__(
        self,
        processing_times: str = "processing_time",
        machine_feature: str = "machine",
        disjunctive: bool = True,
    ):
        self.processing_times = processing_times
        self.machine_feature = machine_feature
        self.disjunctive = disjunctive

    def initialize(self, instance: ProblemInstance) -> None:
        machine_ids = instance.register_task_feature(self.machine_feature)
        p_times = instance.register_task_feature(self.processing_times)

        for task_id, p_time in enumerate(p_times):
            machine_id = MachineID(machine_ids[task_id])

            instance.set_processing_time(task_id, machine_id, Time(p_time))

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        task_disjunction = NonOverlapConstraint(task_groups=instance.job_tasks)

        return (
            (MachineConstraint(), task_disjunction)
            if self.disjunctive
            else (task_disjunction,)
        )

    def get_entry(self) -> str:
        return f"Om"


class JobShopSetup(OpenShopSetup):
    """
    Job Shop Scheduling Setup.

    This setup is used for scheduling tasks in a job shop environment where each task has a specific
    operation order and is assigned to a specific machine.
    """

    operation_order: str

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

        self.operation_order = operation_order

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        precedence_mapping: dict[Int, list[Int]] = {}
        task_orders: list[list[int]] = [[] for _ in range(instance.n_jobs)]

        operations = instance.register_task_feature(self.operation_order)

        for task_id, operation in enumerate(operations):
            job_id = instance.job_ids[task_id]

            if len(task_orders[job_id]) <= operation:
                task_orders[job_id].extend(
                    -1 for _ in range(len(task_orders[job_id]), operation + 1)
                )

            task_orders[job_id][operation] = task_id

        for tasks in task_orders:
            if len(tasks) < 2:
                continue

            prec = tasks[0]
            for task_id in tasks[1:]:
                precedence_mapping[task_id] = [prec]

                prec = task_id

        precedence_constraint = PrecedenceConstraint(precedence_mapping)

        return (
            (MachineConstraint(), precedence_constraint)
            if self.disjunctive
            else (precedence_constraint,)
        )

    def get_entry(self) -> str:
        return f"Jm"


class FlexibleJobShopSetup(UnrelatedParallelMachineSetup):
    """
    Flexible Job Shop Scheduling Setup.

    This setup is a variant of the job shop scheduling problem, where each task
    can be processed on one of several machines instead of a specific machine.
    The operation order is still fixed, but the machine assignment is flexible,
    allowing for more scheduling options and potentially better solutions.
    """

    operation_order: str

    def __init__(
        self,
        processing_times: Iterable[str],
        operation_order: str = "operation",
        disjunctive: bool = True,
    ):
        super().__init__(
            processing_times=processing_times,
            disjunctive=disjunctive,
        )

        self.operation_order = operation_order

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        precedence_mapping: dict[Int, list[Int]] = {}
        task_orders: list[list[int]] = [[] for _ in range(instance.n_jobs)]

        operations = instance.register_task_feature(self.operation_order)

        for task_id, operation in enumerate(operations):
            job_id = instance.job_ids[task_id]

            if len(task_orders[job_id]) <= operation:
                task_orders[job_id].extend(
                    -1 for _ in range(len(task_orders[job_id]), operation + 1)
                )

            task_orders[job_id][operation] = task_id

        for tasks in task_orders:
            if len(tasks) < 2:
                continue

            prec = tasks[0]
            for task_id in tasks[1:]:
                precedence_mapping[task_id] = [prec]

                prec = task_id

        precedence_constraint = PrecedenceConstraint(precedence_mapping)

        return (
            (MachineConstraint(), precedence_constraint)
            if self.disjunctive
            else (precedence_constraint,)
        )

    def get_entry(self) -> str:
        return "FJm"


class FlowShopSetup(JobShopSetup):
    """
    Flow Shop Scheduling Setup.

    This setup is used for scheduling tasks in a flow shop environment where
    each task has a specific operation order and all tasks follow the same
    machine order.
    """

    def __init__(
        self,
        processing_times: str = "processing_time",
        operation_order: str = "operation",
        disjunctive: bool = True,
    ):
        super().__init__(
            processing_times=processing_times,
            operation_order=operation_order,
            machine_feature=operation_order,
            disjunctive=disjunctive,
        )

    def get_entry(self) -> str:
        return f"Fm"

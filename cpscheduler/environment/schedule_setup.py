"""
schedule_setup.py

This module defines the ScheduleSetup class and its subclasses for different scheduling setups.
It provides a framework for creating various scheduling environments, such as single machine,
identical parallel machines, uniform parallel machines, job shop, and open shop setups.
"""

from collections.abc import Iterable

from mypy_extensions import mypyc_attr

from cpscheduler.utils.list_utils import convert_to_list

from cpscheduler.environment.constants import MachineID, Time, Int
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.constraints import (
    Constraint,
    NonOverlapConstraint,
    PrecedenceConstraint,
    MachineConstraint,
)

setups: dict[str, type["ScheduleSetup"]] = {}


def ceil_div(a: Time, b: Time) -> Time:
    "a divided by b, rounded up to the nearest integer."
    return -(-a // b)


@mypyc_attr(allow_interpreted_subclasses=True)
class ScheduleSetup:
    """
    Base class for scheduling setups. It defines the common interface for all scheduling setups
    and provides methods to parse process times, set tasks, and setup constraints.
    """

    n_machines: int = 0

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        setups[cls.__name__] = cls

    def initialize(self, state: ScheduleState) -> None:
        "Initialize the state with the given schedule setup."
        raise NotImplementedError()

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        "Build the constraint for that setup."
        return ()

    def get_entry(self) -> str:
        "Produce the α entry for the constraint."
        return ""


class SingleMachineSetup(ScheduleSetup):
    """
    Single Machine Scheduling Setup.

    This setup is used for scheduling tasks on a single machine.
    """

    n_machines: int = 1

    def __init__(
        self,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ) -> None:
        self.processing_times = processing_times
        self.disjunctive = disjunctive

    def initialize(self, state: ScheduleState) -> None:
        instance = state.instance

        for task_id, p_time in enumerate(
            instance.task_instance[self.processing_times]
        ):
            instance.set_processing_time(task_id, 0, Time(p_time))

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        if not self.disjunctive:
            return ()

        return (MachineConstraint(),)


class IdenticalParallelMachineSetup(ScheduleSetup):
    """
    Identical Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines using the same processing time.
    """

    def __init__(
        self,
        n_machines: int,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.n_machines = n_machines
        self.processing_times = processing_times
        self.disjunctive = disjunctive

    def initialize(self, state: ScheduleState) -> None:
        instance = state.instance

        for task_id, p_time in enumerate(
            instance.task_instance[self.processing_times]
        ):
            for machine in range(self.n_machines):
                instance.set_processing_time(task_id, machine, Time(p_time))

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()


class UniformParallelMachineSetup(ScheduleSetup):
    """
    Uniform Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines with different speeds.
    """

    def __init__(
        self,
        speed: Iterable[Int],
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.speed = convert_to_list(speed, int)

        self.processing_times = processing_times
        self.disjunctive = disjunctive
        self.n_machines = len(self.speed)

    def initialize(self, state: ScheduleState) -> None:
        instance = state.instance

        for task_id, p_time in enumerate(
            instance.task_instance[self.processing_times]
        ):
            for machine, speed in enumerate(self.speed):
                p_time = ceil_div(Time(p_time), speed)

                instance.set_processing_time(task_id, machine, p_time)

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()


class UnrelatedParallelMachineSetup(ScheduleSetup):
    processing_times: list[str]

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

        self.n_machines = len(self.processing_times)

    def initialize(self, state: ScheduleState) -> None:
        instance = state.instance

        for machine, p_time_feature in enumerate(self.processing_times):
            p_times = instance.task_instance[p_time_feature]

            for task_id, p_time in enumerate(p_times):
                instance.set_processing_time(task_id, machine, Time(p_time))

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()


class JobShopSetup(ScheduleSetup):
    """
    Job Shop Scheduling Setup.

    This setup is used for scheduling tasks in a job shop environment where each task has a specific
    operation order and is assigned to a specific machine.
    """

    def __init__(
        self,
        processing_times: str = "processing_time",
        operation_order: str = "operation",
        machine_feature: str = "machine",
    ):
        self.processing_times = processing_times
        self.operation_order = operation_order
        self.machine_feature = machine_feature

    def initialize(self, state: ScheduleState) -> None:
        instance = state.instance

        n_machines = 0

        for task_id in range(instance.n_tasks):
            machine: MachineID = instance.task_instance[self.machine_feature][
                task_id
            ]
            p_time = Time(
                instance.task_instance[self.processing_times][task_id]
            )

            instance.set_processing_time(task_id, machine, p_time)

            if machine >= n_machines:
                n_machines = machine + 1

        self.n_machines = n_machines

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        disjunctive_constraint = MachineConstraint()

        precedence_mapping: dict[Int, list[Int]] = {}
        task_orders: list[list[int]] = [[] for _ in range(state.n_jobs)]

        operations = state.instance.task_instance[self.operation_order]

        for task_id, operation in enumerate(operations):
            job_id = state.instance.job_ids[task_id]

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
                precedence_mapping[prec] = [task_id]

                prec = task_id

        precedence_constraint = PrecedenceConstraint(precedence_mapping)

        return (disjunctive_constraint, precedence_constraint)


class OpenShopSetup(ScheduleSetup):
    """
    Open Shop Scheduling Setup.

    This setup is used for scheduling tasks in an open shop environment where each task can be
    processed on any machine, and the order of operations is not fixed.
    """

    processing_times: list[str]

    def __init__(
        self,
        processing_times: Iterable[str],
        disjunctive: bool = True,
    ):
        self.processing_times = list(processing_times)
        self.disjunctive = disjunctive

        self.n_machines = len(self.processing_times)

    def initialize(self, state: ScheduleState) -> None:
        instance = state.instance

        for machine, p_time_feature in enumerate(self.processing_times):
            p_times = instance.task_instance[p_time_feature]

            for task_id, p_time in enumerate(p_times):
                instance.set_processing_time(
                    task_id, MachineID(machine), Time(p_time)
                )

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        task_disjunction = NonOverlapConstraint(state.instance.job_tasks)

        if not self.disjunctive:
            return (task_disjunction,)

        machine_disjunction = MachineConstraint()

        return (task_disjunction, machine_disjunction)

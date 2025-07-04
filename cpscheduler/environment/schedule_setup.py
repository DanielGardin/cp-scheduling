"""
schedule_setup.py

This module defines the ScheduleSetup class and its subclasses for different scheduling setups.
It provides a framework for creating various scheduling environments, such as single machine,
identical parallel machines, uniform parallel machines, job shop, and open shop setups.
"""

from typing import Any, SupportsInt
from collections.abc import Iterable, Mapping

from abc import ABC, abstractmethod

from mypy_extensions import mypyc_attr

from ._common import ProcessTimeAllowedTypes, MACHINE_ID, TIME
from .tasks import Tasks
from .constraints import (
    Constraint,
    DisjunctiveConstraint,
    PrecedenceConstraint,
    MachineConstraint,
)
from .utils import is_iterable_type, convert_to_list, is_iterable_int

PTIME_ALIASES = ["processing_time", "process_time", "processing time"]

setups: dict[str, type["ScheduleSetup"]] = {}


# TODO: Expand the inference logic to handle more complex cases
def infer_processing_time(data: dict[str, list[Any]]) -> ProcessTimeAllowedTypes:
    for alias in PTIME_ALIASES:
        if alias in data:
            return alias

    raise ValueError(f"Cannot infer processing time from the data.")


@mypyc_attr(allow_interpreted_subclasses=True)
class ScheduleSetup(ABC):
    """
    Base class for scheduling setups. It defines the common interface for all scheduling setups
    and provides methods to parse process times, set tasks, and setup constraints.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        setups[cls.__name__] = cls

    @abstractmethod
    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        """
        Parse the process time of the tasks. The process time can be a list of dictionaries, a
        dictionary of lists, or a pandas DataFrame. The function will return a list of dictionaries
        with the machine as key and the process time as value.

        Parameters:
        data (dict): Dictionary containing the data of the tasks.
        process_time (ProcessTimeAllowedTypes): Process time of the tasks.

        Returns:
        list[dict[MACHINE_ID, TIME]]: List of dictionaries with the machine as key and the process time
        as value.
        """

    def setup_constraints(self, tasks: Tasks) -> tuple[Constraint, ...]:
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

    def __init__(self, disjunctive: bool = True) -> None:
        self.disjunctive = disjunctive

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(data)

        if is_iterable_int(process_time):
            return [{0: TIME(p_time)} for p_time in process_time]

        if isinstance(process_time, str):
            return [{0: TIME(p_time)} for p_time in data[process_time]]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def setup_constraints(self, tasks: Tasks) -> tuple[Constraint, ...]:
        if not self.disjunctive:
            return ()

        disjunctive_tasks = {0: [i for i in range(tasks.n_tasks)]}

        return (DisjunctiveConstraint(disjunctive_tasks, name="disjunctive"),)

    def get_entry(self) -> str:
        return "1"


class IdenticalParallelMachineSetup(ScheduleSetup):
    """
    Identical Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines using the same processing time.
    """

    def __init__(
        self,
        n_machines: int,
        disjunctive: bool = True,
    ):
        self.n_machines = n_machines
        self.disjunctive = disjunctive

    def setup_constraints(self, tasks: Tasks) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
        )

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(data)

        if is_iterable_int(process_time):
            return [
                {
                    MACHINE_ID(machine): TIME(p_time)
                    for machine in range(self.n_machines)
                }
                for p_time in process_time
            ]

        if isinstance(process_time, str):
            return [
                {
                    MACHINE_ID(machine): TIME(p_time)
                    for machine in range(self.n_machines)
                }
                for p_time in data[process_time]
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return f"P{self.n_machines}" if self.n_machines > 1 else "Pm"


class UniformParallelMachineSetup(ScheduleSetup):
    """
    Uniform Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines with different speeds.
    """

    def __init__(
        self,
        speed: Iterable[SupportsInt],
        disjunctive: bool = True,
    ):
        self.speed = convert_to_list(speed, int)

        self.disjunctive = disjunctive
        self.n_machines = len(self.speed)

    def setup_constraints(self, tasks: Tasks) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
        )

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(data)

        if is_iterable_int(process_time):
            return [
                {
                    MACHINE_ID(machine): TIME(p_time) // self.speed[machine]
                    for machine in range(self.n_machines)
                }
                for p_time in process_time
            ]

        if isinstance(process_time, str):
            return [
                {
                    machine: p_time // self.speed[machine]
                    for machine in range(self.n_machines)
                }
                for p_time in data[process_time]
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return f"U{self.n_machines}" if self.n_machines > 1 else "Um"


class UnrelatedParallelMachineSetup(ScheduleSetup):
    def __init__(
        self,
        disjunctive: bool = True,
    ):
        self.disjunctive = disjunctive

    def setup_constraints(self, tasks: Tasks) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
        )

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            raise ValueError(
                "Cannot infer processing time for unrelated parallel machines."
            )

        if isinstance(process_time, str):
            raise ValueError(
                "Unrelated parallel machine setup requires one processing time for each machine."
            )

        if is_iterable_type(process_time, str):
            processing_times = zip(*[data[feat] for feat in process_time])

            return [
                {
                    MACHINE_ID(machine): TIME(p_time)
                    for machine, p_time in enumerate(p_times)
                }
                for p_times in processing_times
            ]

        if is_iterable_int(process_time):
            raise ValueError(
                "Unrelated parallel machine setup requires one processing time for each machine."
            )

        if is_iterable_type(process_time, Mapping):
            return [
                {
                    MACHINE_ID(machine): TIME(p_time)
                    for machine, p_time in p_times.items()
                }
                for p_times in process_time
            ]

        return [
            {
                MACHINE_ID(machine): TIME(p_time)
                for machine, p_time in enumerate(p_times)
            }
            for p_times in process_time
        ]

    def get_entry(self) -> str:
        return "Rm"


class JobShopSetup(ScheduleSetup):
    """
    Job Shop Scheduling Setup.

    This setup is used for scheduling tasks in a job shop environment where each task has a specific
    operation order and is assigned to a specific machine.
    """

    def __init__(
        self,
        operation_order: str = "operation",
        machine_feature: str = "machine",
    ):
        self.operation_order = operation_order
        self.machine_feature = machine_feature

    def setup_constraints(self, tasks: Tasks) -> tuple[Constraint, ...]:
        # disjunctive_constraint = DisjunctiveConstraint(
        #     self.machine_feature, name="setup_disjunctive"
        # )
        disjunctive_constraint = MachineConstraint(name="setup_machine_disjunctive")

        edges: list[tuple[int, int]] = []

        operations: list[int] = tasks.data[self.operation_order]

        for job_tasks in tasks.jobs:
            ops = sorted(
                [(operations[task.task_id], task.task_id) for task in job_tasks]
            )

            for i in range(len(ops) - 1):
                edges.append((ops[i][1], ops[i + 1][1]))

        precedence_constraint = PrecedenceConstraint.from_edges(
            edges, name="setup_precedence"
        )

        return (disjunctive_constraint, precedence_constraint)

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(data)

        if is_iterable_int(process_time):
            return [
                {MACHINE_ID(machine): TIME(p_time)}
                for machine, p_time in zip(data[self.machine_feature], process_time)
            ]

        if isinstance(process_time, str):
            return [
                {MACHINE_ID(machine): TIME(p_time)}
                for machine, p_time in zip(
                    data[self.machine_feature], data[process_time]
                )
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return "Jm"


class OpenShopSetup(ScheduleSetup):
    """
    Open Shop Scheduling Setup.

    This setup is used for scheduling tasks in an open shop environment where each task can be
    processed on any machine, and the order of operations is not fixed.
    """

    def __init__(
        self,
        machine_feature: str = "machine",
        disjunctive: bool = True,
    ):
        self.machine_feature = machine_feature
        self.disjunctive = disjunctive

    def setup_constraints(self, tasks: Tasks) -> tuple[Constraint, ...]:
        task_jobs = {
            job: [int(task.task_id) for task in tasks]
            for job, tasks in enumerate(tasks.jobs)
        }
        task_disjunction = DisjunctiveConstraint(
            task_jobs, name="setup_task_disjunctive"
        )

        if not self.disjunctive:
            return (task_disjunction,)

        machine_disjunction = DisjunctiveConstraint(
            self.machine_feature, name="setup_machine_disjunctive"
        )

        return (task_disjunction, machine_disjunction)

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(data)

        if is_iterable_int(process_time):
            return [
                {MACHINE_ID(machine): TIME(p_time)}
                for machine, p_time in zip(data[self.machine_feature], process_time)
            ]

        if isinstance(process_time, str):
            return [
                {MACHINE_ID(machine): TIME(p_time)}
                for machine, p_time in zip(
                    data[self.machine_feature], data[process_time]
                )
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return "Om"

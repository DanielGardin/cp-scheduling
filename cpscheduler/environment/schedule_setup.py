"""
    schedule_setup.py

    This module defines the ScheduleSetup class and its subclasses for different scheduling setups.
    It provides a framework for creating various scheduling environments, such as single machine,
    identical parallel machines, uniform parallel machines, job shop, and open shop setups.
"""
from typing import Any, ClassVar, Iterable

from abc import ABC, abstractmethod

from mypy_extensions import mypyc_attr

from .common import ProcessTimeAllowedTypes
from .tasks import Tasks
from .constraints import Constraint, DisjunctiveConstraint, PrecedenceConstraint, MachineConstraint
from .utils import is_iterable_type, convert_to_list

@mypyc_attr(allow_interpreted_subclasses=True)
class ScheduleSetup(ABC):
    """
        Base class for scheduling setups. It defines the common interface for all scheduling setups
        and provides methods to parse process times, set tasks, and setup constraints.
    """
    tasks: Tasks
    parallel: ClassVar[bool] = True

    def __init__(
        self,
        n_machines: int = -1,
    ):
        self.n_machines = n_machines

    @abstractmethod
    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        """
        Parse the process time of the tasks. The process time can be a list of dictionaries, a
        dictionary of lists, or a pandas DataFrame. The function will return a list of dictionaries
        with the machine as key and the process time as value.

        Parameters:
        data (dict): Dictionary containing the data of the tasks.
        process_time (ProcessTimeAllowedTypes): Process time of the tasks.

        Returns:
        list[dict[int, int]]: List of dictionaries with the machine as key and the process time
        as value.
        """

    def set_tasks(self, tasks: Tasks) -> None:
        "Make the setup aware of the tasks it is applied to."
        self.tasks = tasks

    def setup_constraints(self) -> tuple[Constraint, ...]:
        "Build the constraint for that setup."
        return ()

    def get_machine(self, task_id: int) -> int:
        "Get the default machine for a given task."
        raise ValueError(
            f"The {self.__class__.__name__} setup does not have a default machine assignment."
        )

    def get_entry(self) -> str:
        "Produce the α entry for the constraint."
        return ""


class SingleMachineSetup(ScheduleSetup):
    """
    Single Machine Scheduling Setup.

    This setup is used for scheduling tasks on a single machine.
    """
    parallel: ClassVar[bool] = False

    def __init__(self, disjunctive: bool = True) -> None:
        super().__init__(1)
        self.disjunctive = disjunctive

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [{0: p_time} for p_time in process_time]

        if isinstance(process_time, str):
            return [{0: p_time} for p_time in data[process_time]]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def setup_constraints(self) -> tuple[Constraint, ...]:
        if not self.disjunctive:
            return ()

        disjunctive_tasks = {0: list(range(len(self.tasks)))}

        return (DisjunctiveConstraint(disjunctive_tasks, name="disjunctive"),)

    def get_machine(self, task_id: int) -> int:
        return 0

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
        super().__init__(n_machines)
        self.disjunctive = disjunctive

    def setup_constraints(self) -> tuple[Constraint, ...]:
        return (MachineConstraint(name="setup_machine_disjunctive"), ) if self.disjunctive else ()

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [
                {machine: p_time for machine in range(self.n_machines)}
                for p_time in process_time
            ]

        if isinstance(process_time, str):
            return [
                {machine: p_time for machine in range(self.n_machines)}
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
        n_machines: int,
        speed: Iterable[float],
        disjunctive: bool = True,
    ):
        super().__init__(n_machines)
        self.speed = convert_to_list(speed, int)
        self.disjunctive = disjunctive

    def setup_constraints(self) -> tuple[Constraint, ...]:
        return (MachineConstraint(name="setup_machine_disjunctive"), ) if self.disjunctive else ()

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [
                {machine: p_time // self.speed[machine] for machine in range(self.n_machines)}
                for p_time in process_time
            ]

        if isinstance(process_time, str):
            return [
                {machine: p_time // self.speed[machine] for machine in range(self.n_machines)}
                for p_time in data[process_time]
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return f"U{self.n_machines}" if self.n_machines > 1 else "Um"

# class UnrelatedParallelMachineSetup(ScheduleSetup):
#     def __init__(
#         self,
#         n_machines: int,
#         disjunctive: bool = True,
#     ):
#         super().__init__(n_machines)
#         self.disjunctive = disjunctive

#     def setup_constraints(self) -> tuple[Constraint, ...]:
#         return (MachineConstraint(name="setup_machine_disjunctive"), ) if self.disjunctive else ()

class JobShopSetup(ScheduleSetup):
    """
    Job Shop Scheduling Setup.

    This setup is used for scheduling tasks in a job shop environment where each task has a specific
    operation order and is assigned to a specific machine.
    """
    parallel: ClassVar[bool] = False

    def __init__(
        self,
        n_machines: int = -1,
        operation_order: str = "operation",
        machine_feature: str = "machine",
    ):
        super().__init__(n_machines)

        self.operation_order = operation_order
        self.machine_feature = machine_feature

    def get_machine(self, task_id: int) -> int:
        machine: int = self.tasks.data[self.machine_feature][task_id]
        return machine

    def setup_constraints(self) -> tuple[Constraint, ...]:
        disjunctive_constraint = DisjunctiveConstraint(
            self.machine_feature,
            name="setup_disjunctive"
        )

        edges: list[tuple[int, int]] = []

        operations: list[int] = self.tasks.data[self.operation_order]

        for job_tasks in self.tasks.jobs:
            ops = sorted(
                [(operations[task.task_id], task.task_id) for task in job_tasks]
            )

            for i in range(len(ops) - 1):
                edges.append(
                    (ops[i][1], ops[i + 1][1])
                )

        precedence_constraint = PrecedenceConstraint.from_edges(edges, name="setup_precedence")

        return (disjunctive_constraint, precedence_constraint)

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [
                {machine: p_time}
                for machine, p_time in zip(data[self.machine_feature], process_time)
            ]

        if isinstance(process_time, str):
            return [
                {machine: p_time}
                for machine, p_time in zip(data[self.machine_feature], data[process_time])
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return f"J{self.n_machines}" if self.n_machines > 1 else "Jm"

class OpenShopSetup(ScheduleSetup):
    """
    Open Shop Scheduling Setup.

    This setup is used for scheduling tasks in an open shop environment where each task can be
    processed on any machine, and the order of operations is not fixed.
    """
    parallel: ClassVar[bool] = False

    def __init__(
        self,
        n_machines: int = -1,
        machine_feature: str = "machine",
        disjunctive: bool = True,
    ):
        super().__init__(n_machines)

        self.machine_feature = machine_feature
        self.disjunctive = disjunctive

    def get_machine(self, task_id: int) -> int:
        machine: int = self.tasks.data[self.machine_feature][task_id]
        return machine

    def setup_constraints(self) -> tuple[Constraint, ...]:
        task_jobs = {
            job: [task.task_id for task in tasks] for job, tasks in enumerate(self.tasks.jobs)
        }
        task_disjunction = DisjunctiveConstraint(task_jobs, name="setup_task_disjunctive")

        if not self.disjunctive:
            return (task_disjunction,)

        machine_disjunction = DisjunctiveConstraint(
            self.machine_feature,
            name="setup_machine_disjunctive"
        )

        return (task_disjunction, machine_disjunction)

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [
                {machine: p_time}
                for machine, p_time in zip(data[self.machine_feature], process_time)
            ]

        if isinstance(process_time, str):
            return [
                {machine: p_time}
                for machine, p_time in zip(data[self.machine_feature], data[process_time])
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return f"O{self.n_machines}" if self.n_machines > 1 else "Om"

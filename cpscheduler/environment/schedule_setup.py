"""
schedule_setup.py

This module defines the ScheduleSetup class and its subclasses for different scheduling setups.
It provides a framework for creating various scheduling environments, such as single machine,
identical parallel machines, uniform parallel machines, job shop, and open shop setups.
"""

from typing import Any
from collections.abc import Iterable, Mapping
from typing_extensions import Self

from mypy_extensions import mypyc_attr

from cpscheduler.utils.list_utils import convert_to_list
from cpscheduler.utils.typing_utils import (
    is_iterable_type,
    is_iterable_int,
    iterate_indexed,
)

from ._common import ProcessTimeAllowedTypes, MACHINE_ID, TIME, Int
from .data import SchedulingData
from .constraints import (
    Constraint,
    DisjunctiveConstraint,
    PrecedenceConstraint,
    MachineConstraint,
)

PTIME_ALIASES = ["processing_time", "process_time", "processing time"]


# TODO: Expand the inference logic to handle more complex cases
def infer_processing_time(data: dict[str, list[Any]]) -> ProcessTimeAllowedTypes:
    for alias in PTIME_ALIASES:
        if alias in data:
            return alias

    raise ValueError(f"Cannot infer processing time from the data.")


setups: dict[str, type["ScheduleSetup"]] = {}


@mypyc_attr(allow_interpreted_subclasses=True)
class ScheduleSetup:
    """
    Base class for scheduling setups. It defines the common interface for all scheduling setups
    and provides methods to parse process times, set tasks, and setup constraints.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        setups[cls.__name__] = cls

    def parse_process_time(
        self,
        task_data: dict[str, list[Any]],
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
        raise NotImplementedError()

    def setup_constraints(self, data: SchedulingData) -> tuple[Constraint, ...]:
        "Build the constraint for that setup."
        return ()

    def get_entry(self) -> str:
        "Produce the α entry for the constraint."
        return ""

    def to_dict(self) -> dict[str, Any]:
        "Serialize the setup to a dictionary."
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        "Deserialize the setup from a dictionary."
        return cls(**data)

    def __reduce__(self) -> tuple[Any, ...]:
        "Support for pickling the setup."
        return (self.__class__, tuple(self.to_dict().values()))


class SingleMachineSetup(ScheduleSetup):
    """
    Single Machine Scheduling Setup.

    This setup is used for scheduling tasks on a single machine.
    """

    def __init__(self, disjunctive: bool = True) -> None:
        self.disjunctive = disjunctive

    def parse_process_time(
        self,
        task_data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(task_data)

        if is_iterable_int(process_time):
            return [{0: TIME(p_time)} for p_time in process_time]

        if isinstance(process_time, str):
            return [{0: TIME(p_time)} for p_time in task_data[process_time]]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def setup_constraints(self, data: SchedulingData) -> tuple[Constraint, ...]:
        if not self.disjunctive:
            return ()

        return (MachineConstraint(name="setup_machine_disjunctive"),)

    def get_entry(self) -> str:
        return "1"

    def to_dict(self) -> dict[str, Any]:
        return {"disjunctive": self.disjunctive}


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

    def setup_constraints(self, data: SchedulingData) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
        )

    def parse_process_time(
        self,
        task_data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(task_data)

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
                for p_time in task_data[process_time]
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return f"P{self.n_machines}" if self.n_machines > 1 else "Pm"

    def to_dict(self) -> dict[str, Any]:
        return {"n_machines": self.n_machines, "disjunctive": self.disjunctive}


class UniformParallelMachineSetup(ScheduleSetup):
    """
    Uniform Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines with different speeds.
    """

    def __init__(
        self,
        speed: Iterable[Int],
        disjunctive: bool = True,
    ):
        self.speed = convert_to_list(speed, int)

        self.disjunctive = disjunctive
        self.n_machines = len(self.speed)

    def setup_constraints(self, data: SchedulingData) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
        )

    def parse_process_time(
        self,
        task_data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(task_data)

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
                for p_time in task_data[process_time]
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return f"U{self.n_machines}" if self.n_machines > 1 else "Um"

    def to_dict(self) -> dict[str, Any]:
        return {"speed": self.speed, "disjunctive": self.disjunctive}


class UnrelatedParallelMachineSetup(ScheduleSetup):
    def __init__(
        self,
        disjunctive: bool = True,
    ):
        self.disjunctive = disjunctive

    def setup_constraints(self, data: SchedulingData) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
        )

    def parse_process_time(
        self,
        task_data: dict[str, list[Any]],
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
            processing_times = zip(*[task_data[feat] for feat in process_time])

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

        return iterate_indexed(process_time)

        # if is_iterable_type(process_time, Mapping):
        #     return [
        #         {
        #             MACHINE_ID(machine): TIME(p_time)
        #             for machine, p_time in p_times.items()
        #         }
        #         for p_times in process_time
        #     ]

        # return [
        #     {
        #         MACHINE_ID(machine): TIME(p_time)
        #         for machine, p_time in enumerate(p_times)
        #     }
        #     for p_times in process_time
        # ]

    def get_entry(self) -> str:
        return "Rm"

    def to_dict(self) -> dict[str, Any]:
        return {"disjunctive": self.disjunctive}


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

    def setup_constraints(self, data: SchedulingData) -> tuple[Constraint, ...]:
        disjunctive_constraint = MachineConstraint(name="setup_machine_disjunctive")

        operations: list[int] = data[self.operation_order]
        precedence_mapping: dict[Int, list[Int]] = {}

        task_orders: list[list[int]] = [[] for _ in range(data.n_jobs)]

        for task_id, (job_id, operation) in enumerate(zip(data.job_ids, operations)):
            if len(task_orders[job_id]) <= operation:
                task_orders[job_id].extend(
                    [-1] * (operation - len(task_orders[job_id]) + 1)
                )

            task_orders[job_id][operation] = task_id

        for tasks in task_orders:
            prec = tasks[0]
            for task_id in tasks[1:]:
                precedence_mapping[prec] = [task_id]

                prec = task_id

        precedence_constraint = PrecedenceConstraint(
            precedence_mapping, name="setup_precedence"
        )

        return (disjunctive_constraint, precedence_constraint)

    def parse_process_time(
        self,
        task_data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(task_data)

        if is_iterable_int(process_time):
            return [
                {MACHINE_ID(machine): TIME(p_time)}
                for machine, p_time in zip(
                    task_data[self.machine_feature], process_time
                )
            ]

        if isinstance(process_time, str):
            return [
                {MACHINE_ID(machine): TIME(p_time)}
                for machine, p_time in zip(
                    task_data[self.machine_feature], task_data[process_time]
                )
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return "Jm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_order": self.operation_order,
            "machine_feature": self.machine_feature,
        }


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

    def setup_constraints(self, data: SchedulingData) -> tuple[Constraint, ...]:
        task_disjunction = DisjunctiveConstraint(
            data.job_ids, name="setup_task_disjunctive"
        )

        if not self.disjunctive:
            return (task_disjunction,)

        machine_disjunction = DisjunctiveConstraint(
            self.machine_feature, name="setup_machine_disjunctive"
        )

        return (task_disjunction, machine_disjunction)

    def parse_process_time(
        self,
        task_data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[MACHINE_ID, TIME]]:
        if process_time is None:
            process_time = infer_processing_time(task_data)

        if is_iterable_int(process_time):
            return [
                {MACHINE_ID(machine): TIME(p_time)}
                for machine, p_time in zip(
                    task_data[self.machine_feature], process_time
                )
            ]

        if isinstance(process_time, str):
            return [
                {MACHINE_ID(machine): TIME(p_time)}
                for machine, p_time in zip(
                    task_data[self.machine_feature], task_data[process_time]
                )
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def get_entry(self) -> str:
        return "Om"

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine_feature": self.machine_feature,
            "disjunctive": self.disjunctive,
        }

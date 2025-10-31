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

from cpscheduler.environment._common import MACHINE_ID, TIME, Int, ceil_div
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.constraints import (
    Constraint,
    DisjunctiveConstraint,
    PrecedenceConstraint,
    MachineConstraint,
)

setups: dict[str, type["ScheduleSetup"]] = {}


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
        "Initialize the setup with the given schedule state."
        raise NotImplementedError()

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
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
    n_machines: int = 1

    def __init__(
        self,
        processing_times: str = "processing_time",
        disjunctive: bool = True
    ) -> None:
        self.processing_times = processing_times
        self.disjunctive = disjunctive

    def initialize(self, state: ScheduleState) -> None:
        for task, p_time in zip(state.tasks, state.instance[self.processing_times]):
            task.set_processing_time(0, TIME(p_time))

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
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
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.n_machines = n_machines
        self.processing_times = processing_times
        self.disjunctive = disjunctive

    def initialize(self, state: ScheduleState) -> None:
        for task, p_time in zip(state.tasks, state.instance[self.processing_times]):
            for machine in range(self.n_machines):
                task.set_processing_time(machine, TIME(p_time))

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
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
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.speed = convert_to_list(speed, int)

        self.processing_times = processing_times
        self.disjunctive = disjunctive
        self.n_machines = len(self.speed)

    def initialize(self, state: ScheduleState) -> None:
        for task, p_time in zip(state.tasks, state.instance[self.processing_times]):
            for machine, speed in enumerate(self.speed):
                p_time = ceil_div(TIME(p_time), speed)

                task.set_processing_time(machine, p_time)

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
        )

    def get_entry(self) -> str:
        return f"U{self.n_machines}" if self.n_machines > 1 else "Um"

    def to_dict(self) -> dict[str, Any]:
        return {"speed": self.speed, "disjunctive": self.disjunctive}


class UnrelatedParallelMachineSetup(ScheduleSetup):
    def __init__(
        self,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.processing_times = processing_times
        self.disjunctive = disjunctive

    def initialize(self, state: ScheduleState) -> None:
        n_machines = 0

        for task, p_times in zip(state.tasks, state.instance[self.processing_times]):
            if isinstance(p_times, Mapping):
                for machine, p_time in p_times.items():
                    task.set_processing_time(MACHINE_ID(machine), TIME(p_time))

                    n_machines = max(n_machines, machine + 1)

            elif is_iterable_type(p_times, int):
                for machine, p_time in enumerate(p_times):
                    task.set_processing_time(MACHINE_ID(machine), TIME(p_time))

                    n_machines = max(n_machines, machine + 1)

            else:
                raise ValueError(
                    "Processing times must be a dictionary or an iterable of integers."
                )

        self.n_machines = n_machines

    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        return (
            (MachineConstraint(name="setup_machine_disjunctive"),)
            if self.disjunctive
            else ()
        )

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
        processing_times: str = "processing_time",
        operation_order: str = "operation",
        machine_feature: str = "machine",
    ):
        self.processing_times = processing_times
        self.operation_order = operation_order
        self.machine_feature = machine_feature


    def initialize(self, state: ScheduleState) -> None:
        n_machines = 0

        for task, p_time, machine in zip(
            state.tasks,
            state.instance[self.processing_times],
            state.instance[self.machine_feature]
        ):
            task.set_processing_time(MACHINE_ID(machine), TIME(p_time))

            n_machines = max(n_machines, int(machine) + 1)

        self.n_machines = n_machines


    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        disjunctive_constraint = MachineConstraint(name="setup_machine_disjunctive")

        operations: list[int] = state.instance[self.operation_order]
        precedence_mapping: dict[Int, list[Int]] = {}

        task_orders: list[list[int]] = [[] for _ in range(state.n_jobs)]


        for task, operation in zip(state.tasks, operations):
            job_id = task.job_id

            while len(task_orders[job_id]) <= operation:
                task_orders[job_id].append(-1)
            
            task_orders[job_id][operation] = task.task_id

        for tasks in task_orders:
            prec = tasks[0]
            for task_id in tasks[1:]:
                precedence_mapping[prec] = [task_id]

                prec = task_id

        precedence_constraint = PrecedenceConstraint(
            precedence_mapping, name="setup_precedence"
        )

        return (disjunctive_constraint, precedence_constraint)

    def get_entry(self) -> str:
        return f"J{self.n_machines}" if self.n_machines > 1 else "Jm"

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
        n_machines: int | None = None,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        self.processing_times = processing_times
        self.disjunctive = disjunctive

        if n_machines is not None:
            self.n_machines = n_machines

    def initialize(self, state: ScheduleState) -> None:
        n_machines = 0

        for task, p_times in zip(state.tasks, state.instance[self.processing_times]):
            if isinstance(p_times, Int):
                for machine in range(self.n_machines):
                    task.set_processing_time(MACHINE_ID(machine), TIME(p_times))

            if isinstance(p_times, Mapping):
                for machine, p_time in p_times.items():
                    task.set_processing_time(MACHINE_ID(machine), TIME(p_time))

                    n_machines = max(n_machines, machine + 1)

            elif is_iterable_type(p_times, int):
                for machine, p_time in enumerate(p_times):
                    task.set_processing_time(MACHINE_ID(machine), TIME(p_time))

                    n_machines = max(n_machines, machine + 1)

            else:
                raise ValueError(
                    "Processing times must be a dictionary or an iterable of integers."
                )

        if self.n_machines > 0:
            self.n_machines = n_machines


    def setup_constraints(self, state: ScheduleState) -> tuple[Constraint, ...]:
        task_disjunction = DisjunctiveConstraint(
            [task.job_id for task in state.tasks],
            name="setup_task_disjunctive"
        )

        if not self.disjunctive:
            return (task_disjunction,)

        machine_disjunction = MachineConstraint(name="setup_machine_disjunctive")

        return (task_disjunction, machine_disjunction)

    def get_entry(self) -> str:
        return f"O{self.n_machines}" if self.n_machines > 1 else "Om"

    def to_dict(self) -> dict[str, Any]:
        return {
            "disjunctive": self.disjunctive,
        }

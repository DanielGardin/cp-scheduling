from collections.abc import Iterable

from cpscheduler.environment.constants import Int, Time
from cpscheduler.environment.constraints import Constraint, MachineConstraint
from cpscheduler.environment.instance import (
    UNSET,
    Feature,
    MachineFeature,
    ProblemInstance,
    TaskFeature,
)
from cpscheduler.environment.setups.base import ScheduleSetup
from cpscheduler.environment.utils.general import convert_to_list


def ceil_div(a: Time, b: Time) -> Time:
    "a divided by b, rounded up to the nearest integer."
    return -(-a // b)


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
            default=(convert_to_list(speed, int) if speed is not None else UNSET),
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

        return "Qm"

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
            name=processing_times,
            elem_type=list[Time],
            shape=("n_machines",),
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
                instance.set_processing_time(task_id, machine_id, ptime)

    def setup_constraints(self, instance: ProblemInstance) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    def get_entry(self) -> str:
        if self.processing_times.loaded:
            return f"R{self.n_machines}"

        return "Rm"

    @classmethod
    def get_general_entry(cls) -> str:
        return "Rm"

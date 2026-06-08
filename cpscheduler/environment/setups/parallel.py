"""Schedule setups for parallel machine scheduling problems.

Parallel machine scheduling problems involve scheduling tasks/jobs over time,
where each task can be processed on one of several machines.
The goal is to optimize a specific objective by selecting the appropriate machine
for each task and determining the order of tasks on each machine.

The setups in this module include:
- SingleMachineSetup: A setup for scheduling tasks on a single machine.
- IdenticalParallelMachineSetup: A setup for scheduling tasks on multiple identical machines.
- UniformParallelMachineSetup: A setup for scheduling tasks on multiple uniform machines with different speeds.
- UnrelatedParallelMachineSetup: A setup for scheduling tasks on multiple unrelated machines with different processing
times for each task-machine pair.

"""

from collections.abc import Iterable
from typing import override

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


def _ceil_div(a: Time, b: Time) -> Time:
    """Return division rounded up to the nearest integer."""
    return -(-a // b)


class SingleMachineSetup(ScheduleSetup):
    """Single Machine Scheduling Setup.

    This setup is used for scheduling tasks that have a single resource (machine)
    available for processing.
    """

    processing_times: TaskFeature[Time]

    disjunctive: bool

    def __init__(
        self,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ) -> None:
        """Initialize the Single Machine Setup.

        Parameters
        ----------
        processing_times: str
            The name of the task feature that contains the processing times.

        disjunctive: bool
            Whether to include disjunctive constraints for the single machine.
            If False, the machine can process multiple tasks simultaneously,
            which is equivalent to having no machine constraints.
            When disjunctive is True, the setup will include constraints to ensure
            that only one task can be processed at a time on the machine.

        """
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times,
            semantic="duration",
            shape=(),
        )

    @property
    @override
    def n_machines(self) -> int:
        return 1

    @override
    def get_features(self) -> list[TaskFeature]:
        return [self.processing_times]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, p_time in enumerate(self.processing_times.value):
            instance.set_processing_time(task_id, 0, p_time)

    @override
    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        if not self.disjunctive:
            return ()

        return (MachineConstraint(),)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "1"


class IdenticalParallelMachineSetup(ScheduleSetup):
    """Identical Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple identical machines, where
    each machine has the same processing time for each task.
    """

    _n_machines: int
    processing_times: TaskFeature[Time]
    disjunctive: bool

    def __init__(
        self,
        n_machines: int,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        """Initialize the Identical Parallel Machine Setup.

        Parameters
        ----------
        n_machines: int
            The number of identical machines.

        processing_times: str
            The name of the task feature that contains the processing times.

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

        self._n_machines = n_machines

    @property
    @override
    def n_machines(self) -> int:
        return self._n_machines

    @override
    def get_features(self) -> list[TaskFeature]:
        return [self.processing_times]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, p_time in enumerate(self.processing_times.value):
            for machine in range(self.n_machines):
                instance.set_processing_time(task_id, machine, p_time)

    @override
    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    @override
    def get_entry(self) -> str:
        return f"P{self.n_machines}"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Pm"


class UniformParallelMachineSetup(ScheduleSetup):
    """Uniform Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple machines, where each machine
    has a different speed, and the processing time of a task on a machine is equal to
    the task's processing time divided by the machine's speed.
    """

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
        """Initialize the Uniform Parallel Machine Setup.

        Parameters
        ----------
        speed: Iterable[Int] | None = None
            An iterable of integers representing the speeds of the machines. If None,
            the speeds will be obtained from the instance after loading.

        speed_tag: str
            The name of the machine feature that contains the machine speeds.
            This is used when the speeds are not provided directly to the constructor, but
            are instead loaded from the instance.

        processing_times: str
            The name of the task feature that contains the processing times.

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

        self.speed = MachineFeature(
            name=speed_tag,
            semantic="discrete",
            default=(
                convert_to_list(speed, int) if speed is not None else UNSET
            ),
        )

    @property
    @override
    def n_machines(self) -> int:
        if self.speed.loaded:
            return len(self.speed.value)

        return 0

    @override
    def get_features(self) -> list[Feature]:
        return [
            self.speed,
            self.processing_times,
        ]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        if any(s <= 0 for s in self.speed.value):
            raise ValueError("Machine speeds must be positive integers.")

        for task_id, p_time in enumerate(self.processing_times.value):
            for machine, speed in enumerate(self.speed.value):
                machine_p_time = _ceil_div(p_time, speed)

                instance.set_processing_time(task_id, machine, machine_p_time)

    @override
    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    @override
    def get_entry(self) -> str:
        if self.speed.loaded:
            return f"Q{self.n_machines}"

        return "Qm"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Qm"


class UnrelatedParallelMachineSetup(ScheduleSetup):
    """Unrelated Parallel Machine Scheduling Setup.

    This setup is used for scheduling tasks on multiple unrelated machines,
    where each task-machine pair has a different processing time, and there is
    no specific relationship between the processing times of different machines
    for the same task.
    """

    processing_times: TaskFeature[list[Time]]
    disjunctive: bool

    def __init__(
        self,
        processing_times: str = "processing_time",
        disjunctive: bool = True,
    ):
        """Initialize the Unrelated Parallel Machine Setup.

        Parameters
        ----------
        processing_times: str
            The name of the task feature that contains the processing times.
            The feature should be a list of lists, where each inner list contains
            the processing times for all machines, for each task.

        disjunctive: bool
            Whether to include disjunctive constraints for the machines.
            If False, machines can process multiple tasks simultaneously.
            When disjunctive is True, the setup will include constraints to ensure
            that only one task can be processed at a time on each machine.

        """
        self.disjunctive = disjunctive

        self.processing_times = TaskFeature(
            name=processing_times,
            shape=("n_machines",),
            semantic="duration",
        )

    @property
    @override
    def n_machines(self) -> int:
        if self.processing_times.loaded:
            return len(self.processing_times.value[0])

        return 0

    @override
    def get_features(self) -> list[TaskFeature]:
        return [self.processing_times]

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, machine_times in enumerate(self.processing_times.value):
            for machine_id, ptime in enumerate(machine_times):
                instance.set_processing_time(task_id, machine_id, ptime)

    @override
    def setup_constraints(
        self, instance: ProblemInstance
    ) -> tuple[Constraint, ...]:
        return (MachineConstraint(),) if self.disjunctive else ()

    @override
    def get_entry(self) -> str:
        if self.processing_times.loaded:
            return f"R{self.n_machines}"

        return "Rm"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "Rm"

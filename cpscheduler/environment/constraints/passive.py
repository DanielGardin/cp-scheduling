"""Passive constraints for the scheduling environment."""

from collections.abc import Iterable, Mapping

from typing_extensions import Self, override

from cpscheduler.environment.constants import Int, MachineID, TaskID, Time
from cpscheduler.environment.constraints.base import PassiveConstraint
from cpscheduler.environment.instance import ProblemInstance


class PreemptionConstraint(PassiveConstraint):
    """Preemption constraint for the scheduling environment.

    This constraint allows tasks to be preempted, meaning they can be interrupted
    and resumed later.
    """

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id in range(instance.n_tasks):
            instance.set_preemption(task_id)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "prmp"


class OptionalityConstraint(PassiveConstraint):
    """Makes tasks optional in the scheduling environment.

    Tasks marked as optional are treated equally to regular tasks, but they can be
    left unscheduled without affecting the feasibility of the overall schedule.
    """

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id in range(instance.n_tasks):
            instance.set_optionality(task_id)

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "opt"


class ConstantProcessingTime(PassiveConstraint):
    """Constant processing time constraint for the scheduling environment.

    This constraint enforces that all tasks have the same processing time.

    """

    processing_time: Time

    def __init__(self, processing_time: Int = 1):
        """Initialize the Constant Processing Time Constraint.

        Parameters
        ----------
        processing_time: Int
            The constant processing time for all tasks.

        """
        self.processing_time = Time(processing_time)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id in range(instance.n_tasks):
            for machine in instance.get_machines(task_id):
                instance.set_processing_time(
                    task_id, machine, self.processing_time
                )

    @override
    def get_entry(self) -> str:
        return f"p_j={self.processing_time}"

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "p_j=p"


class MachineEligibilityConstraint(PassiveConstraint):
    """Machine eligibility constraint for the scheduling environment.

    This constraint defines the machines on which each task can be executed.

    This constraint is limited by the setup of the scheduling environment,
    meaning that you cannot:
    - add machines that do not exist in the environment.
    - include/exclude machines that would make the task incompatible with the scheduling setup.
    - exclude all machines for a task.

    By default, if eligibility is not defined for a task, it is assumed that the task
    can be executed on the original set of machines defined by the scheduling setup.

    """

    eligibility: dict[TaskID, set[MachineID]]

    def __init__(self, eligibility: Mapping[Int, Iterable[Int]] | None = None):
        """Initialize the Machine Eligibility Constraint.

        Parameters
        ----------
        eligibility: Mapping[Int, Iterable[Int]] | None
            A mapping where the keys are task IDs and the values are iterable
            of machine IDs that are eligible to process the corresponding task.
            If None, all tasks are assumed to be eligible on all machines defined
            in the scheduling setup.

        """
        if eligibility is None:
            eligibility = {}

        self.eligibility = {
            TaskID(task): {MachineID(machine) for machine in machines}
            for task, machines in eligibility.items()
        }

    @classmethod
    def from_mask(cls, mask: list[list[bool]]) -> Self:
        """Create a MachineEligibilityConstraint from a binary mask of shape (n_tasks, n_machines)."""
        eligibility: Mapping[Int, Iterable[Int]] = {
            task_id: [
                m_id for m_id, eligible in enumerate(machine_mask) if eligible
            ]
            for task_id, machine_mask in enumerate(mask)
        }

        return cls(eligibility)

    def add_eligibility(self, task_id: Int, machine_id: Int) -> None:
        """Add eligibility for a task on a specific machine."""
        if TaskID(task_id) not in self.eligibility:
            self.eligibility[TaskID(task_id)] = set()

        self.eligibility[TaskID(task_id)].add(MachineID(machine_id))

    def remove_eligibility(self, task_id: Int, machine_id: Int) -> None:
        """Remove eligibility for a task on a specific machine."""
        if TaskID(task_id) in self.eligibility:
            self.eligibility[TaskID(task_id)].discard(MachineID(machine_id))

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for task_id, allowed in self.eligibility.items():
            current = set(instance.get_machines(task_id))

            for machine_id in current - allowed:
                instance.remove_machine(task_id, machine_id)

            if not instance.get_machines(task_id):
                raise ValueError(f"Task {task_id} has no eligible machines.")

    @classmethod
    @override
    def get_general_entry(cls) -> str:
        return "M_j"

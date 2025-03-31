from typing import ClassVar

from textwrap import dedent

from .constraints import Constraint, DisjunctiveConstraint, PrecedenceConstraint
from .tasks import Tasks


class ScheduleSetup:
    parallel_machines: ClassVar[bool] = True

    tasks: Tasks

    def __init__(
        self,
        n_machines: int = -1,
    ):
        self._machines = n_machines

    @property
    def n_machines(self) -> int:
        if self._machines <= 0:
            return self.get_n_machines()

        return self._machines

    def get_n_machines(self) -> int:
        """
        Get the number of machines in the setup at runtime. This is useful for setups that
        do not have a fixed number of machines at initialization, for instance Pm.
        """
        raise NotImplementedError(
            "The number of machines must be defined in the general setup"
        )

    def set_tasks(self, tasks: Tasks) -> None:
        self.tasks = tasks

    def setup_constraints(self) -> tuple[Constraint, ...]:
        return ()

    def get_machine(self, task_id: int) -> int:
        raise NotImplementedError(
            "There's no default machine assignment in the general setup"
        )

    def export_model(self) -> str:
        model = """
            % (Schedule Setup) Task identical machines processing time
            constraint forall(t in 1..num_tasks)(
                sum(p in 1..num_parts)(duration[t,p]) = processing_time[t]
            );
        """

        return dedent(model)

    def export_data(self) -> str:
        return ""

    def get_entry(self) -> str:
        return ""


class SingleMachineSetup(ScheduleSetup):
    parallel_machines: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__(1)

    def setup_constraints(self) -> tuple[Constraint, ...]:
        disjunctive_tasks = {0: list(range(len(self.tasks)))}

        return DisjunctiveConstraint(disjunctive_tasks),

    def get_machine(self, task_id: int) -> int:
        return 1

    def get_entry(self) -> str:
        return "1"


class JobShopSetup(ScheduleSetup):
    parallel_machines: ClassVar[bool] = False

    def __init__(
        self,
        n_machines: int = -1,
        operation_order: str = "operation",
        machine_feature: str = "machine",
    ):
        super().__init__(n_machines)

        self.operation_order = operation_order
        self.machine_feature = machine_feature

    def get_n_machines(self) -> int:
        machines: set[int] = {machine for machine in self.tasks.data[self.machine_feature]}
        return len(machines)

    def get_machine(self, task_id: int) -> int:
        machine: int = self.tasks.data[self.machine_feature][task_id]
        return machine

    def setup_constraints(self) -> tuple[Constraint, ...]:
        disjunctive_constraint = DisjunctiveConstraint(self.machine_feature)

        precedence_tasks: dict[int, list[int]] = {}

        operations: list[int] = self.tasks.data[self.operation_order]

        for job_tasks in self.tasks.jobs.values():
            ops = sorted(
                [(operations[task.task_id], task.task_id) for task in job_tasks]
            )

            for i in range(len(ops) - 1):
                precedence_tasks[ops[i][1]] = [ops[i + 1][1]]

        precedence_constraint = PrecedenceConstraint(precedence_tasks)

        return disjunctive_constraint, precedence_constraint

    def get_entry(self) -> str:
        return f"J{self.n_machines}" if self._machines > 1 else "Jm"

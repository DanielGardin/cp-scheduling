from typing import ClassVar, Optional

from textwrap import dedent

from .constraints import Constraint, DisjunctiveConstraint, PrecedenceConstraint
from .tasks import Tasks


class ScheduleSetup:
    parallel_machines: ClassVar[bool] = False
    n_machines: int

    tasks: Tasks

    def __init__(
        self,
        n_machines: Optional[int] = None,
    ):
        self.n_machines = -1 if n_machines is None else n_machines

    def get_n_machines(self) -> int:
        raise NotImplementedError(
            "The number of machines must be defined in the general setup"
        )

    def set_tasks(self, tasks: Tasks) -> None:
        self.tasks = tasks

        if self.n_machines == -1:
            self.n_machines = self.get_n_machines()

    def setup_constraints(self) -> tuple[Constraint, ...]:
        return ()

    def get_machine(self, task_id: int) -> int:
        raise NotImplementedError(
            "There's no default machine assignment in the general setup"
        )

    def export_model(self) -> str:
        model = """
            constraint forall(t in 1..num_tasks)(
                sum(p in 1..num_parts)(duration[t,p]) = processing_time[t]
            );
        """

        return dedent(model)

    def export_data(self) -> str:
        return ""


class JobShopSetup(ScheduleSetup):
    parallel_machines: ClassVar[bool] = False

    def __init__(
        self,
        operation_order: str = "operation",
        machine_feature: str = "machine",
    ):
        super().__init__()

        self.operation_order = operation_order
        self.machine_feature = machine_feature

    def get_n_machines(self) -> int:
        return max(self.tasks.data[self.machine_feature], default=0)

    def get_machine(self, task_id: int) -> int:
        machine: int = self.tasks.data[self.machine_feature][task_id]
        return machine

    def setup_constraints(self) -> tuple[Constraint, ...]:
        disjunctive_tasks: dict[int, list[int]] = {}

        machines: list[int] = self.tasks.data[self.machine_feature]
        for task_id, machine in enumerate(machines):
            if machine not in disjunctive_tasks:
                disjunctive_tasks[machine] = []

            disjunctive_tasks[machine].append(task_id)

        disjunctive_constraint = DisjunctiveConstraint(disjunctive_tasks)

        precedence_tasks: dict[int, list[int]] = {}

        operations: list[int] = self.tasks.data[self.operation_order]

        for job, job_tasks in self.tasks.jobs.items():
            ops = sorted(
                [(operations[task.task_id], task.task_id) for task in job_tasks]
            )

            for i in range(len(ops) - 1):
                precedence_tasks[ops[i][1]] = [ops[i + 1][1]]

        precedence_constraint = PrecedenceConstraint(precedence_tasks)

        return disjunctive_constraint, precedence_constraint

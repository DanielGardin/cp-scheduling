from collections.abc import Iterable, Sequence
from abc import ABC, abstractmethod

from pulp import LpVariable, LpProblem, LpBinary, LpAffineExpression, lpSum

from cpscheduler.environment.tasks import Tasks

from .pulp_utils import implication_pulp

def get_order(task_i: int, task_j: int) -> tuple[tuple[int, int], int]:
    "Util function to get the proper order of tasks in the ordering variable."
    if task_i < task_j:
        return (task_i, task_j), 1
    else:
        return (task_j, task_i), 0

class PulpVariables(ABC):
    def __init__(self, model: LpProblem, tasks: Tasks, integral: bool):
        self.model    = model
        self.tasks    = tasks
        self.integral = integral

    @property
    @abstractmethod
    def end_times(self) -> Sequence[LpVariable | LpAffineExpression]:
        """
        Expression for the end time of a task.
        Returns:
            A list of LpAffineExpression representing the end times of tasks.
        """

    @abstractmethod
    def get_assignments(self) -> list[tuple[int, int]]:
        """
        Get the assignments of tasks to machines.
        Returns:
            A list of tuples where each tuple contains the machine ID and the start time of the task.
        """

    def set_objective(self, objective_var: LpVariable | LpAffineExpression) -> None:
        self.objective = objective_var

    def get_objective_value(self) -> float:
        if hasattr(self, 'objective'):
            obj_value = self.objective.value()

            if obj_value is None:
                raise ValueError("Objective variable has not been set or is None.")

            return float(obj_value)

        raise ValueError("Objective variable has not been set.")


class PulpSchedulingVariables(PulpVariables):
    start_times: list[LpVariable]
    _end_times: list[LpVariable]
    assignments: list[list[LpVariable]]
    orders: dict[tuple[int, int], LpVariable]

    def __init__(self, model: LpProblem, tasks: Tasks, integral: bool):
        super().__init__(model, tasks, integral)

        # By considering the start and end times as continuous variables,
        # we can get a speedup in the branch-and-bound algorithm.
        self.start_times = [
            LpVariable(
                f"start_{task.task_id}",
                lowBound=task.get_start_lb(),
                upBound=task.get_start_ub(),
                # cat=LpInteger
            ) for task in tasks
        ]

        self._end_times = [
            LpVariable(
                f"end_{task.task_id}",
                lowBound=task.get_end_lb(),
                upBound=task.get_end_ub(),
                # cat=LpInteger
            ) for task in tasks
        ]

        self.assignments = [[
                LpVariable(
                    f"assign_{task.task_id}_{machine_id}",
                    cat=LpBinary
                ) for machine_id in range(tasks.n_machines)
            ] for task in tasks
        ]

        self.orders = {
            (i, j) : LpVariable(
                f"order_{i}_{j}",
                lowBound=0,
                upBound=1,
                cat=LpBinary
            ) for j in range(tasks.n_tasks) for i in range(j)
        }

    @property
    def end_times(self) -> list[LpVariable]:
        return self._end_times

    def get_assignments(self) -> list[tuple[int, int]]:
        assignments: list[tuple[int, int]] = []

        for task_id, task in enumerate(self.tasks):
            start_value = self.start_times[task_id].value()

            start_time = int(start_value) if start_value is not None else -1
            machine_id = -1
            for machine_id in range(self.tasks.n_machines):
                if self.assignments[task_id][machine_id].value() == 1:
                    break

            assignments.append((machine_id, start_time))

        return assignments

    def set_order(
            self,
            model: LpProblem,
            task_prec: int,
            task_succ: int,
            conditions: Iterable[LpVariable] | None = None,
            prefix: str = "",
        ) -> None:
        """
        Set the order of two tasks based on the conditions provided.
        If the conditions are met, the order variable is set to 1, otherwise it is set to 0.
        """
        order, value = get_order(task_prec, task_succ)

        if conditions is None:
            model.addConstraint(
                self.end_times[task_prec] <= self.start_times[task_succ],
                f"{prefix}_C_{task_prec}_le_S_{task_succ}"
            )

            model.addConstraint(
                self.orders[order] == value,
                f"{prefix}_{task_prec}_prec_{task_succ}"
            )

        else:
            implication_pulp(
                model,
                antecedent=conditions,
                consequent = (self.end_times[task_prec], "<=", self.start_times[task_succ]),
                big_m      = int(
                    self.tasks[task_prec].get_end_ub() - self.tasks[task_succ].get_start_lb()
                ),
                name=f"{prefix}_{task_prec}_prec_{task_succ}"
            )

            implication_pulp(
                model,
                antecedent=conditions,
                consequent=(self.orders[order], "==", value),
                big_m=1,
                name=f"{prefix}_{task_prec}_prec_{task_succ}"
            )

class PulpTimetable(PulpVariables):
    T: int
    start_times: list[dict[int, list[LpVariable]]]

    def __init__(self, model: LpProblem, tasks: Tasks, integral: bool):
        super().__init__(model, tasks, integral)

        self.T = tasks.get_time_ub() + 1
        self.start_times = [{
                machine_id: [
                    LpVariable(
                        name=f"start_{task_id}_{machine_id}_{time}",
                        lowBound=0,
                        upBound=1,
                        cat=LpBinary
                    ) for time in range(task.get_start_lb(machine_id), task.get_start_ub(machine_id) + 1)
                ]
                for machine_id in task.machines
            }
            for task_id, task in enumerate(tasks)
        ]


        for task_id in range(tasks.n_tasks):
            starting_times = self.start_times[task_id].values()

            model.addConstraint(
                lpSum(
                    decision_var
                    for machine_decision in starting_times
                    for decision_var in machine_decision
                ) == 1
            )

    def x(self, task_id: int, machine_id: int, time: int) -> LpVariable | int:
        start_lb: int = self.tasks[task_id].get_start_lb(machine_id)
        start_ub: int = self.tasks[task_id].get_start_ub(machine_id)

        if time < start_lb or time >= start_ub:
            return 0

        return self.start_times[task_id][machine_id][time - start_lb]

    @property
    def end_times(self) -> list[LpVariable | LpAffineExpression]:
        "Expression for the end time of a task."
        return [
            lpSum(
            x * (time + self.tasks[task_id].processing_times[machine_id])
            for machine_id, machine_xs in self.start_times[task_id].items()
            for time, x in enumerate(machine_xs, start=self.tasks[task_id].get_start_lb(machine_id))
        ) for task_id in range(self.tasks.n_tasks)]

    def get_assignments(self) -> list[tuple[int, int]]:
        assignments: list[tuple[int, int]] = []

        for task_id in range(self.tasks.n_tasks):
            task = self.tasks[task_id]

            for machine_id in task.machines:
                for time in range(task.get_start_lb(machine_id), task.get_start_ub(machine_id) + 1):
                    if self.start_times[task_id][machine_id][time].value():
                        assignments.append((machine_id, time))

        return assignments
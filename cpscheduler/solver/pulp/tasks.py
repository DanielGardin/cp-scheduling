from typing import Any
from collections.abc import Sequence, Iterable, Mapping
from abc import ABC, abstractmethod

from pulp import (
    LpVariable,
    LpProblem,
    LpBinary,
    LpInteger,
    LpContinuous,
    LpAffineExpression,
    lpSum,
)

from cpscheduler.environment.tasks import Tasks
from cpscheduler.environment.data import SchedulingData

from .pulp_utils import PULP_EXPRESSION, get_value, pulp_add_constraint


def count_variables(variables: Iterable[Any] | PULP_EXPRESSION | int) -> int:
    if isinstance(variables, Mapping):
        return sum(count_variables(v) for v in variables.values())

    if isinstance(variables, Iterable):
        return sum(count_variables(v) for v in variables)

    if isinstance(variables, (LpVariable, LpAffineExpression)):
        return 1

    return 0


class PulpVariables(ABC):
    def __init__(
        self, model: LpProblem, tasks: Tasks, data: SchedulingData, integral: bool
    ):
        object.__setattr__(self, "_initializing_base", True)
        self._variables: dict[str, Any] = {}

        self.model = model
        self.n_tasks = tasks.n_tasks
        self.n_machines = data.n_machines

        self.integral = integral
        self._initializing_base = False

        self.objective: PULP_EXPRESSION = LpAffineExpression()

    def __setattr__(self, name: str, value: Any) -> None:
        if not self._initializing_base and not hasattr(self, name):
            self._variables[name] = value

        super().__setattr__(name, value)

    @property
    def n_variables(self) -> int:
        "Get the number of variables in the model."
        return count_variables(self._variables)

    @property
    @abstractmethod
    def end_times(self) -> Sequence[PULP_EXPRESSION | int]:
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

    def set_objective(self, objective_var: PULP_EXPRESSION) -> None:
        self.objective = objective_var

    def get_objective_value(self) -> float:
        if hasattr(self, "objective"):
            obj_value = self.objective.value()

            if obj_value is None:
                raise ValueError("Objective variable has not been set or is None.")

            return float(obj_value)

        return 0.0


class PulpSchedulingVariables(PulpVariables):
    start_times: list[PULP_EXPRESSION | int]
    _end_times: list[PULP_EXPRESSION | int]
    assignments: list[list[PULP_EXPRESSION | int]]
    orders: dict[tuple[int, int], PULP_EXPRESSION | int]

    def __init__(
        self,
        model: LpProblem,
        tasks: Tasks,
        data: SchedulingData,
        integral: bool,
        integral_var: bool,
    ):
        super().__init__(model, tasks, data, integral)

        # By considering the start and end times as continuous variables,
        # we can get a speedup in the branch-and-bound algorithm.
        self.start_times = [
            LpVariable(
                f"start_{task.task_id}",
                lowBound=task.get_start_lb(),
                upBound=task.get_start_ub(),
                cat=LpInteger if integral_var else LpContinuous,
            )
            for task in tasks
        ]

        self._end_times = [
            LpVariable(
                f"end_{task.task_id}",
                lowBound=task.get_end_lb(),
                upBound=task.get_end_ub(),
                cat=LpInteger if integral_var else LpContinuous,
            )
            for task in tasks
        ]

        self.assignments = []
        assignments: list[PULP_EXPRESSION | int] = []
        for task in tasks:
            machines = task.machines

            if len(machines) == 1:
                assignments = [
                    1 if machine_id == machines[0] else 0
                    for machine_id in range(data.n_machines)
                ]

            else:
                assignments = [0] * data.n_machines

                for machine_id in machines:
                    assignments[machine_id] = LpVariable(
                        f"assign_{task.task_id}_{machine_id}", cat=LpBinary
                    )

            self.assignments.append(assignments)

        self.orders = {}

    @property
    def end_times(self) -> list[PULP_EXPRESSION | int]:
        return self._end_times

    def get_assignments(self) -> list[tuple[int, int]]:
        assignments: list[tuple[int, int]] = []

        for task_id in range(self.n_tasks):
            start_value = get_value(self.start_times[task_id])

            start_time = round(start_value) if start_value is not None else -1
            machine_id = -1
            for machine_id in range(self.n_machines):
                if get_value(self.assignments[task_id][machine_id]) > 1 / len(
                    self.assignments[task_id]
                ):
                    break

            assignments.append((machine_id, start_time))

        return assignments

    def get_order(self, task_prec: int, task_succ: int) -> PULP_EXPRESSION | int:
        if task_prec < task_succ:
            i, j = task_prec, task_succ
            ordered = True

        else:
            i, j = task_succ, task_prec
            ordered = False

        if (i, j) not in self.orders:
            self.orders[(i, j)] = LpVariable(
                f"order_{i}_{j}", lowBound=0, upBound=1, cat=LpBinary
            )

        return self.orders[(i, j)] if ordered else 1 - self.orders[(i, j)]

    def __repr__(self) -> str:
        "String representation of the PulpSchedulingVariables."
        return f"PulpSchedulingVariables(n_variables={self.n_variables})"


class PulpTimetable(PulpVariables):
    T: int
    start_times: list[dict[int, list[PULP_EXPRESSION | int]]]

    def __init__(
        self, model: LpProblem, tasks: Tasks, data: SchedulingData, integral: bool
    ):
        super().__init__(model, tasks, data, integral)

        self.T = tasks.get_time_ub() + 1

        self.start_times = []
        self.processing_times = data.processing_times.copy()

        for task_id, task in enumerate(tasks):
            machines = task.machines

            task_start_times: dict[int, list[PULP_EXPRESSION | int]] = {}
            for machine in machines:
                start_lb = task.get_start_lb(machine)
                start_ub = task.get_start_ub(machine)

                task_start_times[machine] = [
                    (
                        0
                        if time < start_lb or time > start_ub
                        else LpVariable(
                            f"start_{task_id}_{machine}_{time}",
                            lowBound=0,
                            upBound=1,
                            cat=LpBinary,
                        )
                    )
                    for time in range(self.T)
                ]

            self.start_times.append(task_start_times)

            pulp_add_constraint(
                model,
                lpSum(
                    decision_var
                    for machine_decision in task_start_times.values()
                    for decision_var in machine_decision
                )
                == 1,
                f"task_{task_id}_timetable_assignment",
            )

    @property
    def end_times(self) -> list[LpVariable | LpAffineExpression]:
        "Expression for the end time of a task."
        return [
            lpSum(
                self.start_times[task_id][machine_id][time]
                * (time + self.processing_times[task_id][machine_id])
                for machine_id, machine_xs in self.start_times[task_id].items()
                for time in range(self.T)
            )
            for task_id in range(self.n_tasks)
        ]

    def get_assignments(self) -> list[tuple[int, int]]:
        assignments: list[tuple[int, int]] = []

        for task_id in range(self.n_tasks):
            for machine_id in range(self.n_machines):
                for time in range(self.T):
                    if get_value(self.start_times[task_id][machine_id][time]):
                        assignments.append((machine_id, time))

        return assignments

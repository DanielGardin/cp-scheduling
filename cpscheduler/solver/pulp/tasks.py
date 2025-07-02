from typing import Literal
from abc import ABC, abstractmethod

from pulp import LpVariable, LpProblem, LpBinary, LpAffineExpression, lpSum

from cpscheduler.environment.tasks import Tasks


class PulpVariables(ABC):
    def __init__(self, model: LpProblem, tasks: Tasks):
        self.model = model
        self.tasks = tasks

    @property
    @abstractmethod
    def end_times(self) -> list[LpVariable | LpAffineExpression]:
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
    def __init__(
            self,
            model: LpProblem,
            tasks: Tasks
        ):
        super().__init__(model, tasks)

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
    def end_times(self) -> list[LpVariable | LpAffineExpression]:
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

class PulpTimetable(PulpVariables):
    def __init__(self, model: LpProblem, tasks: Tasks):
        super().__init__(model, tasks)

        self.T = tasks.get_time_ub() + 1
        self.start_times: list[dict[int, list[LpVariable]]] = [{
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
        start_lb = self.tasks[task_id].get_start_lb(machine_id)
        start_ub = self.tasks[task_id].get_start_ub(machine_id)

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


    # def get_assignments(self) -> list[tuple[int, int]]:
    #     assignments: list[tuple[int, int]] = []

    #     for task_id in range(self.tasks.n_tasks):


    #         for machine_id in range(self.tasks.n_machines):
    #             for time in range(len(self.start_times)):

    #                 if [machine_id][time].value() == 1:
    #                     assignments.append((machine_id, time))

    #     return assignments
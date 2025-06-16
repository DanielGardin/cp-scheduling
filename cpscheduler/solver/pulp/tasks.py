from abc import ABC, abstractmethod

from pulp import LpVariable, LpInteger, LpBinary, LpAffineExpression

from cpscheduler.environment.tasks import Tasks


class PulpVariables(LpVariable, ABC):
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
            tasks: Tasks
        ):
        self.tasks = tasks

        self.start_times = [
            LpVariable(
                f"start_{task.task_id}",
                lowBound=task.get_start_lb(),
                upBound=task.get_start_ub(),
                cat=LpInteger
            ) for task in tasks
        ]

        self.end_times = [
            LpVariable(
                f"end_{task.task_id}",
                lowBound=task.get_end_lb(),
                upBound=task.get_end_ub(),
                cat=LpInteger
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
    def __init__(self, tasks: Tasks):
        self.tasks = tasks

        time_ub = tasks.get_time_ub()

        self.starting_times: list[list[list[LpVariable]]] = [[[
                    LpVariable(
                        name=f"start_{task_id}_{machine_id}_{time}",
                        lowBound=0,
                        upBound=1,
                        cat=LpBinary
                    ) for task_id in range(tasks.n_tasks)
                ] for machine_id in range(tasks.n_machines)
            ] for time in range(time_ub)
        ]


    def get_assignments(self) -> list[tuple[int, int]]:
        assignments: list[tuple[int, int]] = []

        for time in range(len(self.starting_times)):
            for machine_id in range(self.tasks.n_machines):
                for task_id in range(self.tasks.n_tasks):
                    if self.starting_times[time][machine_id][task_id].value() == 1:
                        assignments.append((machine_id, time))

        return assignments
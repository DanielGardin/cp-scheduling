from pulp import LpVariable, LpInteger, LpBinary, value, LpAffineExpression

from cpscheduler.environment.tasks import Tasks

class PulpVariables:
    start_times: list[LpVariable]
    end_times: list[LpVariable]
    assignments: list[list[LpVariable]]
    orders: dict[tuple[int, int], LpVariable]
    objective: LpVariable | LpAffineExpression

    def __init__(
            self,
            tasks: Tasks,
            n_machines: int,
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
                ) for machine_id in range(n_machines)
            ] for task in tasks
        ]

        self.orders = {
            (i, j) : LpVariable(
                f"order_{i}_{j}",
                lowBound=0,
                upBound=1,
                cat=LpBinary
            ) for i in range(tasks.n_tasks) for j in range(i)
        }
    
    def get_start_times(self) -> list[int]:
        start_times = map(value, self.start_times)

        return [int(start_time) for start_time in start_times]

    def get_end_times(self) -> list[int]:
        end_times = map(value, self.end_times)

        return [int(end_time) for end_time in end_times]

    def get_assignments(self) -> list[list[int]]:
        assignments = [
            [int(value(assignment)) for assignment in task_assignments] # type: ignore[no-untyped-call]
            for task_assignments in self.assignments
        ]

        return assignments

    def get_orders(self) -> dict[tuple[int, int], int]:
        orders = {
            (i, j): int(value(order)) # type: ignore
            for (i, j), order in self.orders.items()
            if value(order) is not None
        }

        return orders

    def set_objective(self, objective_var: LpVariable | LpAffineExpression) -> None:
        self.objective = objective_var

    def get_objective_value(self) -> float:
        if hasattr(self, 'objective'):
            obj_value = value(self.objective)

            if obj_value is None:
                raise ValueError("Objective variable has not been set or is None.")

            assert isinstance(obj_value, (float, int))
            return float(obj_value)

        raise ValueError("Objective variable has not been set.")

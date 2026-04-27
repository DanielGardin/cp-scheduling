from typing import Any

from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.state import ScheduleState

from cpscheduler.solver.cp.minizinc_formulation import MiniZincFormulation

class DisjunctiveCPFormulation(MiniZincFormulation):
    formulation_name = "disjunctive_cp"

    start_times: list[int | str]
    end_times: list[int | str]
    assignments: list[list[str | bool]]
    presents: list[str | bool]
    orders: dict[tuple[int, int], str]
    _warm_start_annotations: list[str]
    _horizon: int

    def __init__(self) -> None:
        super().__init__()

        self.start_times = []
        self.end_times = []
        self.assignments = []
        self.presents = []
        self.orders = {}
        self._warm_start_annotations = []
        self._horizon = 0

    def _expr(self, value: int | str | bool) -> str:
        return "true" if value is True else "false" if value is False else str(value)

    def _present_expr(self, task_id: int) -> str:
        return self._expr(self.presents[task_id])

    def _start_expr(self, task_id: int) -> str:
        return self._expr(self.start_times[task_id])

    def _end_expr(self, task_id: int) -> str:
        return self._expr(self.end_times[task_id])

    def _assignment_expr(self, task_id: int, machine_id: int) -> str:
        return self._expr(self.assignments[task_id][machine_id])

    def _duration_expr(self, state: Any, task_id: int) -> str:
        schedule_state = state.state if hasattr(state, "state") else state

        terms: list[str] = []
        for machine_id in range(schedule_state.n_machines):
            processing_time = schedule_state.get_remaining_time(task_id, machine_id)
            if processing_time == 0:
                continue

            assignment = self._assignment_expr(task_id, machine_id)
            terms.append(f"bool2int({assignment}) * {processing_time}")

        return " + ".join(terms) if terms else "0"

    def _solve_annotation(self) -> str:
        if not self._warm_start_annotations:
            return ""

        return " :: warm_start_array([" + ", ".join(
            self._warm_start_annotations
        ) + "])"

    def warm_start(self, env: SchedulingEnv) -> None:
        state: ScheduleState = env.state

        int_vars: list[tuple[str, int]] = []
        bool_vars: list[tuple[str, bool]] = []

        def add_int(var: int | str, value: int) -> None:
            if isinstance(var, str):
                int_vars.append((var, value))

        def add_bool(var: str | bool, value: bool) -> None:
            if isinstance(var, str):
                bool_vars.append((var, value))

        for task_id in range(state.n_tasks):
            if not state.is_fixed(task_id):
                continue

            start_time = state.runtime.get_start(task_id)
            end_time = state.runtime.get_end(task_id)
            machine_id = state.runtime.get_assignment(task_id)

            add_int(self.start_times[task_id], start_time)
            add_int(self.end_times[task_id], end_time)

            for m_id, assignment in enumerate(self.assignments[task_id]):
                add_bool(assignment, m_id == machine_id)

            add_bool(self.presents[task_id], True)

        for (i, j), order_var in self.orders.items():
            if not (state.is_fixed(i) and state.is_fixed(j)):
                continue

            end_i = state.runtime.get_end(i)
            start_j = state.runtime.get_start(j)
            add_bool(order_var, end_i <= start_j)

        annotations: list[str] = []

        if int_vars:
            names = ", ".join(var for var, _ in int_vars)
            values = ", ".join(str(value) for _, value in int_vars)
            annotations.append(
                f"warm_start(array1d(1..{len(int_vars)}, [{names}]), "
                f"array1d(1..{len(int_vars)}, [{values}]))"
            )

        if bool_vars:
            names = ", ".join(var for var, _ in bool_vars)
            values = ", ".join(
                "true" if value else "false" for _, value in bool_vars
            )
            annotations.append(
                f"warm_start(array1d(1..{len(bool_vars)}, [{names}]), "
                f"array1d(1..{len(bool_vars)}, [{values}]))"
            )

        self._warm_start_annotations = annotations

    def initialize_model(self, env: SchedulingEnv) -> None:
        state = env.state

        self.initialize_minizinc_model(minimize=env.objective.minimize)

        self.start_times.clear()
        self.end_times.clear()
        self.assignments.clear()
        self.presents.clear()
        self.orders.clear()

        self._horizon = max(
            1,
            sum(
                max(
                    state.get_remaining_time(task_id, machine_id)
                    for machine_id in state.get_machines(task_id)
                )
                for task_id in range(state.n_tasks)
            ),
        )

        n_machines = state.n_machines

        for task_id in range(state.n_tasks):
            machines = list(state.get_machines(task_id))
            optional = state.is_optional(task_id)

            if state.is_fixed(task_id):
                start_time = state.runtime.get_start(task_id)
                end_time = state.runtime.get_end(task_id)
                machine_id = state.runtime.get_assignment(task_id)

                self.start_times.append(start_time)
                self.end_times.append(end_time)
                self.presents.append(True)
                self.assignments.append([
                    machine == machine_id
                    for machine in range(n_machines)
                ])
                continue

            start_lb = 0 if optional else state.get_start_lb(task_id)
            end_lb = 0 if optional else state.get_end_lb(task_id)

            start_var = self.add_int_var(
                f"start_{task_id}",
                lb=start_lb,
                ub=min(state.get_start_ub(task_id), self._horizon),
            )
            end_var = self.add_int_var(
                f"end_{task_id}",
                lb=end_lb,
                ub=min(state.get_end_ub(task_id), self._horizon),
            )

            self.start_times.append(start_var.name)
            self.end_times.append(end_var.name)

            if state.is_present(task_id):
                present_expr: str | bool = True
            elif state.is_absent(task_id):
                present_expr = False
            elif optional:
                present_var = self.add_bool_var(f"present_{task_id}")
                present_expr = present_var.name
            else:
                present_expr = True

            self.presents.append(present_expr)

            assignment_arr = self.add_bool_var_array(f"assign_{task_id}", n_machines)
            task_assignments: list[str | bool] = []

            for machine_id in range(n_machines):
                expr = assignment_arr[machine_id + 1]
                if machine_id not in machines:
                    self.add_constraint(
                        f"({expr}) = false",
                        f"assign_{task_id}_{machine_id}_forbid",
                    )
                    task_assignments.append(False)
                    continue

                task_assignments.append(expr)

            self.assignments.append(task_assignments)

            assignment_sum = " + ".join(
                f"bool2int({self._assignment_expr(task_id, machine_id)})"
                for machine_id in range(n_machines)
            )

            if present_expr is True:
                self.add_constraint(
                    f"{assignment_sum} = 1",
                    f"assignment_{task_id}",
                )
            elif present_expr is False:
                self.add_constraint(
                    f"{assignment_sum} = 0",
                    f"assignment_{task_id}",
                )
                self.add_constraint(
                    f"{self._start_expr(task_id)} = 0",
                    f"absent_start_{task_id}",
                )
                self.add_constraint(
                    f"{self._end_expr(task_id)} = 0",
                    f"absent_end_{task_id}",
                )
            else:
                self.add_constraint(
                    f"{assignment_sum} = bool2int({present_expr})",
                    f"assignment_{task_id}",
                )
                self.add_constraint(
                    f"{present_expr} -> ({self._start_expr(task_id)} >= {state.get_start_lb(task_id)})",
                    f"present_start_lb_{task_id}",
                )
                self.add_constraint(
                    f"{present_expr} -> ({self._end_expr(task_id)} >= {state.get_end_lb(task_id)})",
                    f"present_end_lb_{task_id}",
                )
                self.add_constraint(
                    f"({present_expr}) = false -> ({self._start_expr(task_id)} = 0)",
                    f"absent_start_{task_id}",
                )
                self.add_constraint(
                    f"({present_expr}) = false -> ({self._end_expr(task_id)} = 0)",
                    f"absent_end_{task_id}",
                )

            self.add_constraint(
                f"{self._end_expr(task_id)} = {self._start_expr(task_id)} + ({self._duration_expr(state, task_id)})",
                f"duration_{task_id}",
            )

    def has_order(self, i: int, j: int) -> bool:
        return tuple(sorted((i, j))) in self.orders

    def get_order(self, i: int, j: int) -> str:
        if i == j:
            return "true"

        key = (i, j) if i < j else (j, i)
        if key not in self.orders:
            order_var = self.add_bool_var(f"order_{key[0]}_{key[1]}")
            self.orders[key] = order_var.name

        expr = self.orders[key]
        return expr if i < j else f"({expr}) = false"

    def set_order(self, i: int, j: int) -> None:
        if i == j:
            return

        self.add_constraint(self.get_order(i, j), f"order_{i}_{j}")

    def get_assignment(self, task_id: int) -> tuple[int, int]:
        start_expr = self.start_times[task_id]
        start_time = (
            int(start_expr)
            if isinstance(start_expr, int)
            else self.get_int_value(start_expr)
        )

        for machine_id, assignment in enumerate(self.assignments[task_id]):
            if isinstance(assignment, bool):
                if assignment:
                    return machine_id, start_time
                continue

            if self.get_bool_value(assignment):
                return machine_id, start_time

        raise ValueError(f"No machine assigned for task {task_id}.")
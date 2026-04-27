from cpscheduler.environment import SchedulingEnv

from cpscheduler.solver.milp.pyomo_formulation import (
    PYOMO_PARAM,
    PyomoFormulation,
)


class DisjunctiveMILPFormulation(PyomoFormulation):
    formulation_name = "disjunctive"

    start_times: list[PYOMO_PARAM]
    end_times: list[PYOMO_PARAM]
    assignments: list[list[PYOMO_PARAM]]
    orders: dict[tuple[int, int], PYOMO_PARAM]
    present: list[PYOMO_PARAM]

    def __init__(self, horizon: int | None = None) -> None:
        super().__init__()

        # Variables
        self.start_times = []
        self.end_times = []
        self.assignments = []
        self.present = []

        self.orders = {}

        self.horizon = horizon

    def get_assignment(self, task_id: int) -> tuple[int, int]:
        start_time = round(self.get_value(self.start_times[task_id]))

        for machine_id, var in enumerate(self.assignments[task_id]):
            if round(self.get_value(var)) == 1:
                return machine_id, start_time

        raise ValueError(f"No machine assigned for task {task_id}.")

    def get_objective_value(self) -> float:
        return self.get_value(self._objective_expr)

    def initialize_model(self, env: SchedulingEnv) -> None:
        self.initialize_pyomo_model(
            name=env.get_entry(),
            minimize=env.objective.minimize,
        )

        state = env.state
        runtime = state.runtime

        n_tasks = state.n_tasks
        n_machines = state.n_machines

        self.start_times = [0] * n_tasks
        self.end_times = [0] * n_tasks
        self.orders.clear()
        self.present = [1] * n_tasks

        self.assignments = [
            [0] * n_machines for _ in range(n_tasks)
        ]

        for task_id in range(n_tasks):
            if state.is_fixed(task_id):
                self.start_times[task_id] = runtime.get_start(task_id)
                self.end_times[task_id] = runtime.get_end(task_id)
                self.assignments[task_id][runtime.get_assignment(task_id)] = 1
                self.present[task_id] = 1
                continue


            self.start_times[task_id] = (
                self.add_var(
                    f"start_time_{task_id}",
                    lb=state.get_start_lb(task_id),
                    ub=state.get_start_ub(task_id),
                )
            )

            self.end_times[task_id] = (
                self.add_var(
                    f"end_time_{task_id}",
                    lb=state.get_end_lb(task_id),
                    ub=state.get_end_ub(task_id),
                )
            )
            presence: PYOMO_PARAM
            if state.is_present(task_id):
                presence = 1

            elif state.is_absent(task_id):
                presence = 0

            else:
                presence = self.add_var(
                    f"present_{task_id}",
                    binary=True,
                )

            machines = state.get_machines(task_id)

            assignments = self.assignments[task_id]
            if len(machines) == 1:
                assignments[machines[0]] = presence

            else:
                for machine_id in machines:
                    assignments[machine_id] = self.add_var(
                        f"assign_{task_id}_{machine_id}",
                        binary=True
                    )


            self.add_constraint(
                sum(assignments[machine_id] for machine_id in machines)
                == presence,
                f"assignment_{task_id}",
            )

            if state.is_preemptive(task_id):
                raise NotImplementedError("Preemptive tasks are not yet supported in the disjunctive formulation.")

            processing_time = sum(
                assignments[machine_id] * int(state.get_remaining_time(task_id, machine_id))
                for machine_id in machines
            )

            self.add_constraint(
                self.end_times[task_id] == self.start_times[task_id] + processing_time,
                f"non_preemptive_{task_id}",
            )

        if self.horizon is not None:
            horizon = self.horizon

            for task_id in range(n_tasks):
                self.add_constraint(
                    self.end_times[task_id] <= horizon,
                    f"horizon_{task_id}"
                )

    def warm_start(self, env: SchedulingEnv) -> None:
        state = env.state
        for task_id in range(state.n_tasks):
            if not state.is_fixed(task_id):
                continue

            start_time = state.runtime.get_start(task_id)
            self.set_initial_value(self.start_times[task_id], start_time)

            end_time = state.runtime.get_end(task_id)
            self.set_initial_value(self.end_times[task_id], end_time)

            machine_id = state.runtime.get_assignment(task_id)
            for m_id, var in enumerate(self.assignments[task_id]):
                self.set_initial_value(var, 1 if m_id == machine_id else 0)

        for (i, j), var in self.orders.items():
            start_i = state.runtime.get_start(i)
            end_i = state.runtime.get_end(i)
            start_j = state.runtime.get_start(j)
            end_j = state.runtime.get_end(j)

            if end_i <= start_j:
                self.set_initial_value(var, 1)

            elif end_j <= start_i:
                self.set_initial_value(var, 0)

        if not env.objective.regular or not state.is_terminal():
            return

        makespan = int(state.runtime.last_completion_time)

        for task_id in range(state.n_tasks):
            if self.get_ub(self.end_times[task_id]) > makespan:
                self.set_ub(self.end_times[task_id], makespan)
                self.set_ub(self.start_times[task_id], makespan)

    # Helper methods for building the model
    def has_order(self, i: int, j: int) -> bool:
        return (i, j) in self.orders

    def set_global_order(self, i: int, j: int) -> None:
        "Unconditionally set i to start before j."

        if i == j:
            return

        if (i, j) not in self.orders:
            self.orders[(i, j)] = 1

        else:
            self.add_constraint(
                self.orders[(i, j)] == 1,
                f"order_{i}_prec_{j}"
            )

        if (j, i) not in self.orders:
            self.orders[(j, i)] = 0
        
        else:
            self.add_constraint(
                self.orders[(j, i)] == 0,
                f"order_{i}_prec_{j}_r"
            )



    def get_order(self, i: int, j: int) -> PYOMO_PARAM:
        if i == j:
            return 1

        if (i, j) in self.orders:
            return self.orders[(i, j)]

        order_var = self.add_var(
            f"order_{i}_{j}",
            binary=True
        )

        self.orders[(i, j)] = order_var

        return order_var

    def set_order(self, i: int, j: int) -> None:

        if (i, j) not in self.orders:
            self.orders[(i, j)] = 1

        else:
            self.add_constraint(self.orders[(i, j)] == 1, f"order_{i}_prec_{j}")

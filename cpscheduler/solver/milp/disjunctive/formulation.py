from typing import Any

from pulp import (
    LpProblem,
    LpSolver,
    LpMinimize,
    LpMaximize,
    LpVariable,
    LpContinuous,
    LpBinary,
    getSolver,
    LpStatus
)

from cpscheduler.environment import SchedulingEnv

from cpscheduler.solver.formulation import Formulation
from cpscheduler.solver.milp.pulp_utils import (
    PULP_PARAM,
    create_binary_var,
    pulp_add_constraint,
    get_value,
    set_initial_value,
    get_ub,
    set_ub,
)


class DisjunctiveMILPFormulation(Formulation, formulation_name="disjunctive"):
    start_times: list[PULP_PARAM]
    end_times: list[PULP_PARAM]
    assignments: list[list[PULP_PARAM]]
    orders: dict[tuple[int, int], PULP_PARAM]
    present: list[PULP_PARAM]

    model: LpProblem

    _solver: LpSolver
    _config: dict[str, Any]

    def __init__(
        self,
        relaxed: bool = False,
        check_symmetries: bool = True,
    ) -> None:
        # Variables
        self.start_times = []
        self.end_times = []
        self.assignments = []
        self.orders = {}
        self.present = []

        self.relaxed = relaxed

        self._config = {}

    def get_assignment(self, task_id: int) -> tuple[int, int]:
        start_time = round(get_value(self.start_times[task_id]))

        for machine_id, var in enumerate(self.assignments[task_id]):
            if get_value(var) == 1:
                return machine_id, start_time

        raise ValueError(f"No machine assigned for task {task_id}.")

    def get_objective_value(self) -> float:
        objective = self.model.objective

        if objective is None:
            return 0.0

        return get_value(objective)

    def initialize_model(self, env: SchedulingEnv) -> None:
        self.start_times.clear()
        self.end_times.clear()
        self.assignments.clear()
        self.orders.clear()
        self.present.clear()

        sense = LpMinimize if env.objective.minimize else LpMaximize
        self.model = LpProblem(env.get_entry(), sense)

        state = env.state
        runtime = state.runtime_state

        n_tasks = state.n_tasks
        n_machines = state.n_machines

        for task_id in range(n_tasks):
            assignments: list[PULP_PARAM] = [0] * n_machines

            if state.is_fixed(task_id):
                self.start_times.append(runtime.get_start(task_id))
                self.end_times.append(runtime.get_end(task_id))
                assignments[runtime.get_assignment(task_id)] = 1
                self.present.append(1)

            else:
                self.start_times.append(
                    LpVariable(
                        f"start_time_{task_id}",
                        lowBound=state.get_start_lb(task_id),
                        upBound=state.get_start_ub(task_id),
                        cat=LpContinuous,
                    )
                )

                self.end_times.append(
                    LpVariable(
                        f"end_time_{task_id}",
                        lowBound=state.get_end_lb(task_id),
                        upBound=state.get_end_ub(task_id),
                        cat=LpContinuous,
                    )
                )

                machines = state.get_machines(task_id)

                if len(machines) == 1:
                    assignments[machines[0]] = 1

                else:
                    for machine_id in machines:
                        assignments[machine_id] = create_binary_var(
                            f"assign_{task_id}_{machine_id}",
                            self.relaxed
                        )

                presence: PULP_PARAM
                if state.is_present(task_id):
                    presence = 1
                
                elif state.is_absent(task_id):
                    presence = 0
                
                else:
                    presence = create_binary_var(
                        f"present_{task_id}",
                        self.relaxed
                    )

                self.present.append(presence)

            self.assignments.append(assignments)

    def solve(
        self,
        solver_tag: str,
        quiet: bool = False,
        time_limit: float | None = None,
        keep_files: bool = False,
        **solver_kwargs: Any
    ) -> str:
        self._solver = getSolver(
            solver_tag,
            msg=not quiet,
            timeLimit=time_limit,
            keepFiles=keep_files,
            **solver_kwargs,
            **self._config
        )

        self.model.solve(self._solver)

        assert isinstance(self.model.status, int)
        status: str = LpStatus[self.model.status]

        return status

    def warm_start(self, env: SchedulingEnv) -> None:
        state = env.state
        for task_id in range(state.n_tasks):
            if not state.is_fixed(task_id):
                continue

            start_time = state.runtime_state.get_start(task_id)
            set_initial_value(self.start_times[task_id], start_time)

            end_time = state.runtime_state.get_end(task_id)
            set_initial_value(self.end_times[task_id], end_time)

            machine_id = state.runtime_state.get_assignment(task_id)
            for m_id, var in enumerate(self.assignments[task_id]):
                set_initial_value(var, 1 if m_id == machine_id else 0)

        for (i, j), var in self.orders.items():
            start_i = state.runtime_state.get_start(i)
            end_i = state.runtime_state.get_end(i)
            start_j = state.runtime_state.get_start(j)
            end_j = state.runtime_state.get_end(j)

            if end_i <= start_j:
                set_initial_value(var, 1)

            elif end_j <= start_i:
                set_initial_value(var, 0)

        self._config["warmStart"] = True

        if not env.objective.regular or not state.is_terminal():
            return

        makespan = int(state.runtime_state.last_completion_time)

        for task_id in range(state.n_tasks):
            if get_ub(self.end_times[task_id]) > makespan:
                set_ub(self.end_times[task_id], makespan)
                set_ub(self.start_times[task_id], makespan)

    # Helper methods for building the model
    def has_order(self, i: int, j: int) -> bool:
        "Check if an order i < j, or j < i exists between two tasks."
        return (i, j) in self.orders or (j, i) in self.orders

    def get_order(self, i: int, j: int) -> PULP_PARAM:
        if i == j:
            return 1

        if i < j:
            if (i, j) not in self.orders:
                self.orders[(i, j)] = LpVariable(
                    f"order_{i}_{j}",
                    lowBound=0,
                    upBound=1,
                    cat=LpContinuous if self.relaxed else LpBinary,
                )

            return self.orders[(i, j)]
    
        if (j, i) not in self.orders:
            self.orders[(j, i)] = LpVariable(
                f"order_{j}_{i}",
                lowBound=0,
                upBound=1,
                cat=LpContinuous if self.relaxed else LpBinary,
            )
        
        return 1 - self.orders[(j, i)]

    def set_order(self, i: int, j: int) -> None:
        if i == j:
            return

        if i < j:
            if (i, j) not in self.orders:
                self.orders[(i, j)] = 1

            else:
                pulp_add_constraint(
                    self.model,
                    self.orders[(i, j)] == 1,
                    f"order_{i}_prec_{j}",
                )
            
        if (j, i) not in self.orders:
            self.orders[(j, i)] = 0

        else:
            pulp_add_constraint(
                self.model,
                self.orders[(j, i)] == 0,
                f"order_{j}_succ_{i}",
            )

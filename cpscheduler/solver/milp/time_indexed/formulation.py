from typing import Any

from pulp import (
    LpProblem,
    LpSolver,
    LpMinimize,
    LpMaximize,
    getSolver,
    LpStatus,
)

from cpscheduler.environment import SchedulingEnv

from cpscheduler.solver.formulation import Formulation
from cpscheduler.solver.milp.pulp_utils import (
    PULP_PARAM,
    create_binary_var,
    get_value,
    set_initial_value,
)


class TimeIndexedMILPFormulation(Formulation):
    formulation_name = "time_indexed"

    starts_at: list[list[PULP_PARAM]]
    assignments: list[list[PULP_PARAM]]
    present: list[PULP_PARAM]

    model: LpProblem

    _solver: LpSolver
    _config: dict[str, Any]

    def __init__(
        self,
        horizon: int,
        relaxed: bool = False,
    ) -> None:
        # Variables
        self.starts_at = []
        self.assignments = []
        self.present = []

        self.horizon = horizon
        self.relaxed = relaxed

        self._config = {}

    def get_assignment(self, task_id: int) -> tuple[int, int]:
        for machine_id, var in enumerate(self.assignments[task_id]):
            if get_value(var) > 0.5:
                assignment = machine_id
                break

        else:
            raise ValueError(f"Task {task_id} is not assigned to any machine.")

        for time, var in enumerate(self.starts_at[task_id]):
            if get_value(var) > 0.5:
                return assignment, time

        raise ValueError(f"Task {task_id} is not scheduled.")

    def get_objective_value(self) -> float:
        objective = self.model.objective

        if objective is None:
            return 0.0

        return get_value(objective)

    def initialize_model(self, env: SchedulingEnv) -> None:
        self.starts_at.clear()
        self.assignments.clear()
        self.present.clear()

        sense = LpMinimize if env.objective.minimize else LpMaximize
        self.model = LpProblem(env.get_entry(), sense)

        state = env.state
        runtime = state.runtime_state

        n_tasks = state.n_tasks
        n_machines = state.n_machines

        for task_id in range(n_tasks):
            assignments: list[PULP_PARAM] = [0] * n_machines
            self.starts_at.append([0] * self.horizon)

            if state.is_fixed(task_id):
                start = runtime.get_start(task_id)
                self.starts_at[task_id][start] = 1

                assignments[runtime.get_assignment(task_id)] = 1
                self.present.append(1)

            else:
                lb = state.get_start_lb(task_id)
                ub = state.get_start_ub(task_id)
                for time in range(lb, ub + 1):
                    self.starts_at[task_id][time] = create_binary_var(
                        f"start_{task_id}_{time}", self.relaxed
                    )

                machines = state.get_machines(task_id)
                if len(machines) == 1:
                    assignments[machines[0]] = 1

                else:
                    for machine_id in machines:
                        assignments[machine_id] = create_binary_var(
                            f"assign_{task_id}_{machine_id}", self.relaxed
                        )

                presence: PULP_PARAM
                if state.is_present(task_id):
                    presence = 1

                elif state.is_absent(task_id):
                    presence = 0

                else:
                    presence = create_binary_var(
                        f"present_{task_id}", self.relaxed
                    )

                self.present.append(presence)

            self.assignments.append(assignments)

    def solve(
        self,
        solver_tag: str,
        quiet: bool = False,
        time_limit: float | None = None,
        keep_files: bool = False,
        **solver_kwargs: Any,
    ) -> str:
        self._solver = getSolver(
            solver_tag,
            msg=not quiet,
            timeLimit=time_limit,
            keepFiles=keep_files,
            **solver_kwargs,
            **self._config,
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

            start = state.runtime_state.get_start(task_id)

            for time, var in enumerate(self.starts_at[task_id]):
                set_initial_value(var, 1 if time == start else 0)

            machine_id = state.runtime_state.get_assignment(task_id)
            for m_id, var in enumerate(self.assignments[task_id]):
                set_initial_value(var, 1 if m_id == machine_id else 0)

        self._config["warmStart"] = True

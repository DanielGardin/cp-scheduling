"""Module for defining the disjunctive MILP formulation."""

from copy import deepcopy

import pyomo.environ as pyo
from typing_extensions import override

from cpscheduler.environment import (
    ComposedObjective,
    Makespan,
    SchedulingEnv,
    SetupConstraint,
)
from cpscheduler.solver.milp.pyomo_formulation import (
    PYOMO_PARAM,
    PyomoFormulation,
)


class DisjunctiveMILPFormulation(PyomoFormulation):
    """A disjunctive MILP formulation for scheduling problems.

    This formulation explicitly models the start and end times of tasks as
    integer variables, and uses binary variables to model the assignment of tasks
    to machines and the ordering of tasks on the same machine.

    The formulation is designed to be flexible and extensible, allowing for
    different types of constraints and objectives to be easily added by registering
    the appropriate functions.

    It may not be the most efficient formulation for all types of scheduling problems,
    specially because it has a weak LP relaxation in general.
    """

    formulation_name = "disjunctive"

    start_times: list[PYOMO_PARAM]
    end_times: list[PYOMO_PARAM]
    assignments: list[list[PYOMO_PARAM]]
    orders: dict[tuple[int, int], PYOMO_PARAM]
    present: list[PYOMO_PARAM]
    _job_presence: dict[int, PYOMO_PARAM]

    def __init__(self, horizon: int | None = None) -> None:
        super().__init__()

        # Variables
        self.start_times = []
        self.end_times = []
        self.assignments = []
        self.present = []
        self._job_presence = {}

        self.orders = {}

        self.horizon = horizon

    @override
    def is_present(self, task_id: int) -> bool:
        return round(self.get_value(self.present[task_id])) == 1

    @override
    def get_assignment(self, task_id: int) -> tuple[int, int]:
        start_time = round(self.get_value(self.start_times[task_id]))

        for machine_id, var in enumerate(self.assignments[task_id]):
            if round(self.get_value(var)) == 1:
                return machine_id, start_time

        raise ValueError(f"No machine assigned for task {task_id}.")

    @override
    def get_objective_value(self) -> float:
        return self.get_value(self._objective_expr)

    @override
    def initialize_model(self, env: SchedulingEnv) -> None:
        super().initialize_model(env)

        state = env.state

        n_tasks = state.n_tasks
        n_machines = state.n_machines

        self.start_times = [0] * n_tasks
        self.end_times = [0] * n_tasks
        self.orders.clear()
        self.present = [1] * n_tasks
        self._job_presence = {}

        self.assignments = [[0] * n_machines for _ in range(n_tasks)]

        for task_id in range(n_tasks):
            if state.is_absent(task_id):
                self.present[task_id] = 0
                continue

            if state.is_fixed(task_id):
                self.start_times[task_id] = state.get_start(task_id)
                self.end_times[task_id] = state.get_end(task_id)
                self.assignments[task_id][state.get_assignment(task_id)] = 1
                self.present[task_id] = 1
                continue

            self.start_times[task_id] = self.add_var(
                f"start_time_{task_id}",
                lb=state.get_start_lb(task_id),
                ub=state.get_start_ub(task_id),
            )

            self.end_times[task_id] = self.add_var(
                f"end_time_{task_id}",
                lb=state.get_end_lb(task_id),
                ub=state.get_end_ub(task_id),
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

            self.present[task_id] = presence

            machines = state.get_machines(task_id)

            assignments = self.assignments[task_id]
            if len(machines) == 1:
                assignments[machines[0]] = presence

            else:
                for machine_id in machines:
                    assignments[machine_id] = self.add_var(
                        f"assign_{task_id}_{machine_id}", binary=True
                    )

            self.add_constraint(
                sum(assignments[machine_id] for machine_id in machines)
                == presence,
                f"assignment_{task_id}",
            )

            processing_time = sum(
                assignments[machine_id]
                * int(state.get_remaining_time(task_id, machine_id))
                for machine_id in machines
            )

            self.add_constraint(
                self.end_times[task_id]
                == self.start_times[task_id] + processing_time,
                f"non_preemptive_{task_id}",
            )

        if self.horizon is None:
            self._tightening_maybe(env)

        else:
            horizon = self.horizon
            for task_id in range(n_tasks):
                self.implication(
                    (self.present[task_id],),
                    (self.end_times[task_id], "<=", horizon),
                    f"horizon_{task_id}",
                )

    def _tightening_maybe(self, env: SchedulingEnv) -> None:
        """Try tightening the upper bounds for a better-behaved model."""
        obj = env.objective
        backend = env.backend

        self._apply_trivial_horizon(env)

        # An incumbent only bounds the end of every task when the objective is
        # the makespan itself. For other objectives, even regular ones, the
        # optimal schedule can finish later than a given feasible schedule.
        if not isinstance(obj, Makespan) or backend.backend != "des":
            return

        cpy_env = deepcopy(env)

        try:
            cpy_env.step(
                [("submit", t) for t in env.state.get_unassigned_tasks()]
            )

        # The greedy schedule is only a hint, and it can get stuck
        except RuntimeError:
            return

        if cpy_env.state.is_terminal() and not cpy_env.state.infeasible:
            self.warm_start(cpy_env)

    # FUTURE: This is an invasive method that should be refactored to a more modular approach,
    # where constraints can register their own tightening methods, for example.
    def _apply_trivial_horizon(self, env: SchedulingEnv) -> None:
        """Bound the time variables, as unbounded ones lead to huge big-M values.

        Numerically, a big-M around the default MAX_TIME is enough for integer
        feasibility tolerances to deactivate the implications.
        """
        state = env.state

        horizon = max(
            (
                int(state.get_start_lb(task_id))
                for task_id in state.get_unassigned_tasks()
            ),
            default=0,
        )
        horizon = max(horizon, int(state.get_latest_end()))

        for task_id in state.get_unassigned_tasks():
            horizon += max(
                int(state.get_remaining_time(task_id, machine_id))
                for machine_id in state.get_machines(task_id)
            )

        for constraint in env.constraints:
            if isinstance(constraint, SetupConstraint):
                horizon += sum(
                    int(setup_time)
                    for setup_times in constraint.setup_times.values()
                    for setup_time in setup_times.values()
                )

        # Non-regular objectives may wait for a due date before starting a task
        objectives = (
            env.objective.objectives
            if isinstance(env.objective, ComposedObjective)
            else [env.objective]
        )
        for objective in objectives:
            due_dates = getattr(objective, "due_dates", None)

            if due_dates is not None and due_dates.value:
                horizon += max(0, *(int(date) for date in due_dates.value))

        for task_id in state.get_unassigned_tasks():
            if self.get_ub(self.end_times[task_id]) > horizon:
                self.set_ub(self.end_times[task_id], horizon)
                self.set_ub(self.start_times[task_id], horizon)

    @override
    def warm_start(self, env: SchedulingEnv) -> None:
        state = env.state
        for task_id in range(state.n_tasks):
            if state.is_absent(task_id):
                self.set_initial_value(self.present[task_id], 0)

                for var in self.assignments[task_id]:
                    self.set_initial_value(var, 0)

                continue

            if not state.is_fixed(task_id):
                continue

            self.set_initial_value(self.present[task_id], 1)

            start_time = state.get_start(task_id)
            self.set_initial_value(self.start_times[task_id], start_time)

            end_time = state.get_end(task_id)
            self.set_initial_value(self.end_times[task_id], end_time)

            machine_id = state.get_assignment(task_id)
            for m_id, var in enumerate(self.assignments[task_id]):
                self.set_initial_value(var, 1 if m_id == machine_id else 0)

        for (i, j), var in self.orders.items():
            # A warm start may be a partial schedule, and rejects tasks
            if not (state.is_assigned(i) and state.is_assigned(j)):
                continue

            start_i = state.get_start(i)
            end_i = state.get_end(i)

            start_j = state.get_start(j)
            end_j = state.get_end(j)

            if end_i <= start_j:
                self.set_initial_value(var, 1)

            elif end_j <= start_i:
                self.set_initial_value(var, 0)

        if not isinstance(env.objective, Makespan) or not state.is_terminal():
            return

        makespan = int(state.get_latest_end())

        for task_id in range(state.n_tasks):
            # A task that may be rejected keeps the bounds of a present one,
            # and its end does not count in the makespan when it is absent.
            if self.is_true(self.present[task_id] == 1) is not True:
                continue

            if self.get_ub(self.end_times[task_id]) > makespan:
                self.set_ub(self.end_times[task_id], makespan)
                self.set_ub(self.start_times[task_id], makespan)

    # Helper methods for building the model
    def job_presence(self, job_id: int, job_tasks: list[int]) -> PYOMO_PARAM:
        """Return whether any task of a job is present."""
        if job_id in self._job_presence:
            return self._job_presence[job_id]

        literals: list[PYOMO_PARAM] = []
        for task_id in job_tasks:
            truth = self.is_true(self.present[task_id] == 1)

            if truth is True:
                self._job_presence[job_id] = 1
                return 1

            if truth is None:
                literals.append(self.present[task_id])

        if not literals:
            self._job_presence[job_id] = 0
            return 0

        presence = self.add_var(f"job_present_{job_id}", binary=True)
        for i, literal in enumerate(literals):
            self.add_constraint(
                presence >= literal, f"job_present_{job_id}_{i}"
            )

        self.add_constraint(
            presence <= pyo.quicksum(literals), f"job_present_{job_id}_any"
        )

        self._job_presence[job_id] = presence
        return presence

    def has_order(self, i: int, j: int) -> bool:
        """Check if there is an order variable defined for tasks i and j."""
        return (i, j) in self.orders

    def set_global_order(self, i: int, j: int) -> None:
        """Unconditionally set i to start before j."""
        if i == j:
            return

        if (i, j) not in self.orders:
            self.orders[(i, j)] = 1

        else:
            self.add_constraint(self.orders[(i, j)] == 1, f"order_{i}_prec_{j}")

        if (j, i) not in self.orders:
            self.orders[(j, i)] = 0

        else:
            self.add_constraint(
                self.orders[(j, i)] == 0, f"order_{i}_prec_{j}_r"
            )

    def get_order(self, i: int, j: int) -> PYOMO_PARAM:
        """Get the order variable for tasks i and j, creating it if it does not exist."""
        if i == j:
            return 1

        if (i, j) in self.orders:
            return self.orders[(i, j)]

        order_var = self.add_var(f"order_{i}_{j}", binary=True)

        self.orders[(i, j)] = order_var

        return order_var

    def set_order(self, i: int, j: int) -> None:
        """Unconditionally set i to start before j, creating the order variable if it does not exist."""
        if (i, j) not in self.orders:
            self.orders[(i, j)] = 1

        else:
            self.add_constraint(self.orders[(i, j)] == 1, f"order_{i}_prec_{j}")

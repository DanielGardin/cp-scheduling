from typing import Any
from collections.abc import Sequence, Iterable, Mapping
from abc import ABC, abstractmethod

from warnings import warn

from pulp import (
    LpVariable,
    LpProblem,
    LpBinary,
    LpInteger,
    LpContinuous,
    LpAffineExpression,
    lpSum,
)

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.env import SchedulingEnv

from .pulp_utils import (
    PULP_EXPRESSION,
    get_value,
    pulp_add_constraint,
    set_initial_value,
    get_initial_value,
)


def count_variables(variables: Iterable[Any] | PULP_EXPRESSION | int) -> int:
    if isinstance(variables, Mapping):
        return sum(count_variables(v) for v in variables.values())

    if isinstance(variables, Iterable):
        return sum(count_variables(v) for v in variables)

    if isinstance(variables, (LpVariable, LpAffineExpression)):
        return 1

    return 0


class PulpVariables(ABC):
    def __init__(self, model: LpProblem, state: ScheduleState, integral: bool):
        object.__setattr__(self, "_initializing_base", True)
        self._variables: dict[str, Any] = {}

        self.model = model
        self.n_tasks = state.n_tasks
        self.n_machines = state.n_machines

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
    def get_assigment(self, task_id: int) -> tuple[int, int]:
        """
        Get the machine assignment for a specific task.
        Args:
            task_id: int
                The ID of the task to get the assignment for.
        Returns:
            A tuple (start_time, machine_id) representing the assignment.
        """

    def set_horizon(self, horizon: int) -> None:
        """
        Set the time horizon for all fixed tasks in the environment.

        Args:
            horizon: int
                The time horizon to be set.
        """
        raise NotImplementedError(f"set_horizon method not implemented for {type(self).__name__}.")

    def warm_start(self, env: SchedulingEnv) -> None:
        """
        Warm start the variables based on the current environment state.
        This method can be overridden by subclasses to implement specific warm start logic.
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
        state: ScheduleState,
        integral: bool,
        integral_var: bool,
    ):
        super().__init__(model, state, integral)

        # By considering the start and end times as continuous variables,
        # we can get a speedup in the branch-and-bound algorithm.
        TimeVarCat = LpInteger if integral_var else LpContinuous

        self.start_times = [
            LpVariable(
                f"start_{task_id}",
                lowBound=state.get_start_lb(task_id),
                upBound=state.get_start_ub(task_id),
                cat=TimeVarCat,
            )
            for task_id in range(state.n_tasks)
        ]

        self._end_times = [
            LpVariable(
                f"end_{task_id}",
                lowBound=state.get_end_lb(task_id),
                upBound=state.get_end_ub(task_id),
                cat=TimeVarCat,
            )
            for task_id in range(state.n_tasks)
        ]

        self.assignments = []
        assignments: list[PULP_EXPRESSION | int] = []
        for task_id in range(state.n_tasks):
            machines = state.tasks[task_id].machines

            assignments = [0] * self.n_machines

            if len(machines) == 1:
                machine = next(iter(machines))
                assignments[machine] = 1

            else:
                for machine in machines:
                    assignments[machine] = LpVariable(f"assign_{task_id}_{machine}", cat=LpBinary)

            self.assignments.append(assignments)

        self.orders = {}

    @property
    def end_times(self) -> list[PULP_EXPRESSION | int]:
        return self._end_times

    def set_horizon(self, horizon: int) -> None:
        """
        Set the time horizon for the scheduling problem.

        Args:
            horizon: int
                The time horizon to be set.
        """
        for start_var, end_var in zip(self.start_times, self._end_times):
            if isinstance(start_var, LpVariable):
                start_var.upBound = horizon

            if isinstance(end_var, LpVariable):
                end_var.upBound = horizon

    def warm_start(self, env: SchedulingEnv) -> None:
        """
        Warm start the variables based on the current environment state.
        This method sets the start times and assignments based on the current schedule.
        """
        makespan = max(env.state.get_end_lb(task_id) for task_id in env.state.fixed_tasks)

        if not env.objective.regular:
            warn(
                f"The objective is not regular and the current makespan of {makespan} may not be a valid upper bound for end times."
                " If this is the case, consider setting a valid upper bound for the end times via `set_horizon`."
            )

        objective = env.get_objective()
        if env.objective.minimize:
            pulp_add_constraint(self.model, self.objective <= objective, "initial_objective_bound")

        else:
            pulp_add_constraint(self.model, self.objective >= objective, "initial_objective_bound")

        if isinstance(self.objective, LpVariable):
            self.objective.setInitialValue(objective, check=False)

        for task_id in env.state.fixed_tasks:
            start_var = self.start_times[task_id]
            end_var = self._end_times[task_id]

            if isinstance(start_var, LpVariable):
                start_time = env.state.get_start_lb(task_id)

                start_var.setInitialValue(start_time, check=False)
                start_var.upBound = makespan

            if isinstance(end_var, LpVariable):
                end_time = env.state.get_end_lb(task_id)

                end_var.setInitialValue(end_time, check=False)
                end_var.upBound = makespan

            machine_assignment = env.state.get_assignment(task_id)
            for machine_id in range(self.n_machines):
                assignment_var = self.assignments[task_id][machine_id]

                set_initial_value(
                    assignment_var,
                    1 if machine_assignment == machine_id else 0,
                    check=False,
                )

        for (task_i, task_j), order in self.orders.items():
            start_i = get_initial_value(self.start_times[task_i])
            start_j = get_initial_value(self.start_times[task_j])

            if start_i <= start_j:
                set_initial_value(order, 1, check=False)

            else:
                set_initial_value(order, 0, check=False)

    def get_assigment(self, task_id: int) -> tuple[int, int]:
        "Get the machine assignment for a specific task."
        machine_id = -1
        for machine_id in range(self.n_machines):
            if get_value(self.assignments[task_id][machine_id]) > 1 / len(
                self.assignments[task_id]
            ):
                break

        start_time = round(get_value(self.start_times[task_id]))

        return machine_id, start_time

    def has_order(self, i: int, j: int) -> bool:
        "Check if an order i < j, or j < i exists between two tasks."
        return (i, j) in self.orders or (j, i) in self.orders

    def get_order(self, task_prec: int, task_succ: int) -> PULP_EXPRESSION | int:
        if task_prec == task_succ:
            return 1

        if task_prec < task_succ:
            i, j = task_prec, task_succ
            ordered = True

        else:
            i, j = task_succ, task_prec
            ordered = False

        if (i, j) not in self.orders:
            self.orders[(i, j)] = LpVariable(f"order_{i}_{j}", lowBound=0, upBound=1, cat=LpBinary)

        return self.orders[(i, j)] if ordered else 1 - self.orders[(i, j)]

    def set_order(self, task_prec: int, task_succ: int) -> None:
        "Add the constraint to enforce the order task_prec < task_succ."
        if task_prec == task_succ:
            return

        if task_prec < task_succ:
            i, j = task_prec, task_succ
            order_value = 1

        else:
            i, j = task_succ, task_prec
            order_value = 0

        if (i, j) not in self.orders:
            self.orders[(i, j)] = order_value

        else:
            order_var = self.orders[(i, j)]

            pulp_add_constraint(
                self.model,
                order_var == order_value,
                f"set_order_{task_prec}_{task_succ}",
            )

    def __repr__(self) -> str:
        "String representation of the PulpSchedulingVariables."
        return f"PulpSchedulingVariables(n_variables={self.n_variables})"


class PulpTimetable(PulpVariables):
    T: int
    start_times: list[dict[int, list[PULP_EXPRESSION | int]]]

    def __init__(self, model: LpProblem, state: ScheduleState, integral: bool):
        super().__init__(model, state, integral)

        self.T = max(state.get_end_ub(task_id) for task_id in range(state.n_tasks))

        self.start_times = []

        for task_id, task in enumerate(state.tasks):
            task_start_times: dict[int, list[PULP_EXPRESSION | int]] = {}

            for machine in task.machines:
                start_lb = state.get_start_lb(task_id, machine)
                start_ub = state.get_start_ub(task_id, machine)

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
        raise NotImplementedError(
            "End times are not directly represented in the timetable formulation."
        )
        # return [
        #     lpSum(
        #         self.start_times[task_id][machine_id][time]
        #         * (time + self.processing_times[task_id][machine_id])
        #         for machine_id, machine_xs in self.start_times[task_id].items()
        #         for time in range(self.T)
        #     )
        #     for task_id in range(self.n_tasks)
        # ]

    def get_assigment(self, task_id: int) -> tuple[int, int]:
        start_time = -1
        machine_id = -1

        for machine_id in range(self.n_machines):
            for time in range(self.T):
                if get_value(self.start_times[task_id][machine_id][time]):
                    start_time = time
                    machine_id = machine_id
                    break

        return start_time, machine_id

"""Naive Constraint Propagation + Branch and Bound solver baseline."""

# import logging
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

from typing_extensions import override

from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.solver.formulation import Formulation


@dataclass
class SolverResults:
    """Search results and statistics."""

    best_solution: float
    optimal: bool

    wall_time: float

    # Search statistics
    nodes: int
    branches: int
    leaves: int
    solutions: int
    infeasible_leaves: int

    pruned_bound: int
    forced_actions: int
    total_events: int

    max_depth: int

    checkpoints: int
    backtracks: int


@dataclass
class Frame:
    """Depth frame storing an environment checkpoint and branches."""

    mark: int
    trace: tuple[TaskID, ...]
    eligible: tuple[TaskID, ...]
    idx: int = 0


def step_into(
    env: SchedulingEnv,
    trace: tuple[TaskID, ...],
) -> tuple[tuple[TaskID, ...], tuple[TaskID, ...], int]:
    """Skip forced actions."""
    forced_actions = 0
    while True:
        eligible = env.get_action_support()

        if not eligible:
            break

        if len(eligible) > 1:
            break

        forced_actions += 1
        action = eligible[0]

        _, _, _, truncated, _ = env.step(("execute", action))
        trace = (*trace, action)

        if truncated:
            eligible = ()
            break

    return trace, tuple(eligible), forced_actions


class NaiveBnBFormulation(Formulation[SolverResults]):
    """Naive branch and bound formulation.

    Employ a DFS search with backtracking and simple pruning.
    Differently from other solvers, NaiveBnBFormulation does not necessarily yields a
    global optimum, but instead, it yields optimal solutions given the
    schedule generation schema the backend implements.

    As this solver steps one action at a time, it relies on backend.eligible_tasks
    to construct a solution, which may not include the optimal solution.
    For example, using the DES backend searches over non-delay schedules, while
    Tetris would search over semi-active schedules, potentially giving different
    optimal solutions for the same instance.
    """

    _env: SchedulingEnv

    best_obj: float
    best_trace: tuple[TaskID, ...]
    solution: dict[TaskID, tuple[MachineID, Time]]
    obj_history: list[float]
    warm_obj: float

    @override
    def initialize_model(self, env: SchedulingEnv) -> None:
        self._env = env
        self.warm_obj = float("inf")

    def solve(
        self,
        quiet: bool = False,
        time_limit: float | None = None,
    ) -> SolverResults:
        """Use branch and bound to solve the environment by single action steps.

        Parameters
        ----------
        quiet: bool, optional
            If True, suppress output. Default is False.

        time_limit: float | None, optional
            Time limit for the solver in seconds. If None, no time limit is applied.

        """
        self.solution = {}
        self.best_trace = ()
        self.obj_history = []
        best_objective = self.warm_obj
        optimal = False

        nodes = 0
        branches = 0
        leaves = 0
        solutions = 0
        infeasible_leaves = 0

        pruned_bound = 0
        max_depth = 0
        backtracks = 0

        env = deepcopy(self._env)
        initial_mark = env.checkpoint()

        start = perf_counter()
        initial_trace, eligible, f_actions = step_into(env, ())
        root_mark = env.checkpoint()
        forced_actions = f_actions
        checkpoints = 1

        stack = [Frame(root_mark + 1, initial_trace, eligible, 0)]

        try:
            time_limit = start + time_limit if time_limit is not None else None

            while stack:
                if time_limit is not None and perf_counter() >= time_limit:
                    break

                frame = stack[-1]

                if frame.idx == len(frame.eligible):
                    env.backtrack(-1)
                    backtracks += 1
                    stack.pop()
                    continue

                nodes += 1
                action = frame.eligible[frame.idx]
                frame.idx += 1

                mark = env.checkpoint()
                checkpoints += 1
                assert mark == frame.mark

                env.step(("execute", action))
                new_trace, eligible, f_actions = step_into(
                    env, (*frame.trace, action)
                )
                forced_actions += f_actions

                if env.state.is_terminal():
                    leaves += 1
                    infeasible_leaves += int(env.state.infeasible)

                    if not env.state.infeasible:
                        solutions += 1
                        obj = env.objective.value

                        if obj < best_objective:
                            best_objective = obj
                            self.obj_history.append(obj)
                            self.best_trace = new_trace

                    env.backtrack(frame.mark)
                    backtracks += 1
                    continue

                if env.objective.lb >= best_objective:
                    pruned_bound += 1

                    env.backtrack(frame.mark)
                    backtracks += 1
                    continue

                if eligible:
                    max_depth = max(max_depth, frame.mark + 1)
                    branches += len(eligible)

                    child = Frame(
                        mark=frame.mark + 1,
                        trace=new_trace,
                        eligible=eligible,
                    )
                    stack.append(child)

            else:
                # Search was exhausted
                optimal = True

        except KeyboardInterrupt:
            pass

        finally:
            end = perf_counter()

        self.best_obj = self.obj_history[-1] if solutions else float("inf")
        total_events = env.event_count

        env.state.finish_propagation()  # Hacky, just to avoid in-propagation errors.
        env.backtrack(initial_mark)
        for action in self.best_trace:
            env.step(("execute", action))

        self.solution = {
            task_id: (
                env.state.get_assignment(task_id),
                env.state.get_start(task_id),
            )
            for task_id in self.best_trace
        }

        return SolverResults(
            best_solution=self.best_obj,
            optimal=optimal,
            wall_time=end - start,
            nodes=nodes,
            branches=branches,
            leaves=leaves,
            solutions=solutions,
            infeasible_leaves=infeasible_leaves,
            pruned_bound=pruned_bound,
            forced_actions=forced_actions,
            total_events=total_events,
            max_depth=max_depth,
            checkpoints=checkpoints,
            backtracks=backtracks,
        )

    @override
    def build(self, env: SchedulingEnv) -> None:
        pass

    @override
    def warm_start(self, env: SchedulingEnv) -> None:
        if env.state.is_terminal():
            self.warm_obj = env.objective.value

    @override
    def get_objective_value(self) -> float:
        return self.best_obj

    @override
    def get_assignment(self, task_id: int) -> tuple[MachineID, Time]:
        return self.solution[task_id]

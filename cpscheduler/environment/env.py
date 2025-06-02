from typing import Any, Optional, Final, Never, ClassVar, overload, Literal
from numpy.typing import NDArray
from pandas import DataFrame

from mypy_extensions import mypyc_attr

import numpy as np
import numpy.lib.recfunctions as rf

from .constraints import Constraint
from .objectives import Objective
from .variables import IntervalVars, Scalar, AVAILABLE_SOLVERS


MIN_INT: Final[int] = -2 ** 31 + 1
MAX_INT: Final[int] =  2 ** 31 - 1


@mypyc_attr(allow_interpreted_subclasses=True)
class SchedulingCPEnv:
    render_mode: ClassVar[list[str]] = ['gantt']

    constraints : dict[str, Constraint]
    objective   : Objective
    minimize    : bool

    current_time    : int
    n_queries       : int
    scheduled_action: NDArray[np.generic]
    current_task    : int

    tasks: IntervalVars


    def __init__(
            self,
            instance: DataFrame,
            duration_feature: str | int
        ) -> None:
        self.constraints = {}
        self.objective   = Objective()
        self.minimize    = True

        tasks = self.process_dataframe(instance)

        self.tasks = IntervalVars(tasks, tasks[duration_feature], instance.index.to_numpy())


    def process_dataframe(self, df: DataFrame, ignore_index: bool = False) -> NDArray[np.void]:
        if ignore_index:
            df.set_index(
                np.arange(len(self.tasks), len(self.tasks) + len(df))
            )

        return np.asarray(df.to_records(index=False))


    def add_constraint(self, constraint: Constraint, name: Optional[str] = None) -> None:
        if name is None: name = constraint.__class__.__name__

        self.constraints[name] = constraint


    def set_objective(self, objective: Objective, minimize: bool = True, name: Optional[str] = None) -> None:
        if name is None: name = objective.__class__.__name__

        self.objective = objective
        self.minimize  = minimize


    def export_model(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        model = [
            self.tasks.export_variables(solver=solver),
            *[constraint.export_constraint(solver=solver) for constraint in self.constraints.values()],
            self.objective.export_objective(self.minimize, solver=solver)
        ]

        return '\n'.join(model)


    def get_cp_solution(
            self,
            timelimit: Optional[float] = None,
            solver: AVAILABLE_SOLVERS = 'cplex'
        ) -> tuple[NDArray[np.int32], NDArray[np.generic], NDArray[np.int32], bool]:
        model_string = self.export_model(solver)

        if solver == 'cplex':
            from docplex.cp.model import CpoModel, CpoSolveResult

            model = CpoModel()

            if timelimit is not None:
                model.set_parameters(TimeLimit=timelimit)

            model.import_model_string(model_string)

            result: CpoSolveResult = model.solve(LogVerbosity="Quiet") # type: ignore

            if result is None:
                raise Exception("No solution found")

            if not result.is_solution():
                raise Exception("No solution found")

            start_times = np.array([
                result.get_var_solution(name).start for name in self.tasks.names # type: ignore
            ], dtype=np.int32)

            objective_values = np.asarray(result.get_objective_values(), dtype=np.int32)
            is_optimal = result.is_solution_optimal()

            order = np.argsort(start_times)


        elif solver == 'ortools':
            from ortools.sat.python import cp_model

            model = cp_model.CpModel()

            locals_: dict[str, cp_model.LinearExprT] = {}

            exec(model_string, {'model' : model}, locals_)

            cp_solver = cp_model.CpSolver()

            if timelimit is not None:
                cp_solver.parameters.max_time_in_seconds = timelimit

            status = cp_solver.Solve(model)

            if status is cp_model.INFEASIBLE: # type: ignore[comparison-overlap]
                raise Exception("No solution found")

            start_times = np.array([
                cp_solver.Value(locals_[f"{name}_start"]) for name in self.tasks.names
            ], dtype=np.int32)

            objective_values = np.array([cp_solver.ObjectiveValue()], dtype=np.int32)
            is_optimal = status is cp_model.OPTIMAL # type: ignore[comparison-overlap]

        else:
            raise ValueError(f"Unknown solver {solver}, available solvers are 'cplex' or 'ortools'")

        order = np.argsort(start_times, stable=True)
        free_tasks = order[~self.tasks.is_fixed()[order]]

        task_order = self.tasks.ids[free_tasks]

        return start_times, task_order, objective_values, is_optimal



    def is_terminal(self) -> bool:
        return bool(self.tasks.is_fixed().all() and (self.current_time >= self.tasks.end_lb[:]).all())


    def is_truncated(self) -> bool:
        return False


    def _get_obs(self) -> NDArray[np.void]:
        obs = self.tasks.get_state(self.current_time)

        indexed_obs = rf.append_fields(
            obs,
            'task_id',
            self.tasks.ids,
            usemask=False
        )

        fields = obs.dtype.names

        if fields is None:
            return np.asarray(rf.unstructured_to_structured(self.tasks.ids))

        indexed_obs = indexed_obs[['task_id', *fields]]

        return np.asarray(indexed_obs)


    def _get_info(self) -> dict[str, Any]:
        info = {
            'n_queries'         : self.n_queries,
            'current_time'      : self.current_time,
            'executed_actions'  : self.scheduled_action[:self.current_task],
            'scheduled_actions' : self.scheduled_action[self.current_task:],
            'objective_value'   : self.objective.get_current()
        }

        if not self.is_terminal(): return info
        
        info |= {
            'solution' : np.argsort(self.tasks.start_lb[:], stable=True).astype(np.int32),
        }

        return info


    def reset(self) -> tuple[NDArray[np.void], dict[str, Any]]:
        self.check_env()

        self.current_time = 0
        self.current_task = 0
        self.scheduled_action = np.array([], dtype=self.tasks.index_dtype)

        self.n_queries = 0

        self.tasks.reset_state()
        
        for constraint in self.constraints.values():
            constraint.reset()

        return self._get_obs(), self._get_info()


    def check_env(self) -> None | Never:
        """
            Check if the constraints and objectives are consistent with the tasks.
        """
        if type(self.objective) is Objective:
            raise ValueError("No objectives have been added to the environment. Please add at least one objective.")
    
        return None


    # TODO: Implement a SAT-CP algorithm for dealing with constraint propagations
    def update_state(self) -> None:
        original = self.tasks.to_propagate.copy()

        for constraint in self.constraints.values():
            constraint.propagate()

        if np.all(original == self.tasks.to_propagate):
            self.tasks.to_propagate[:] = False
            return

        while np.any(self.tasks.to_propagate):
            self.tasks.to_propagate[:] = False

            for constraint in self.constraints.values():
                constraint.propagate()

        self.tasks.to_propagate[:] = False


    def step(
            self,
            actions: Optional[NDArray[np.generic]]       = None,
            time_skip: Optional[int]                     = None,
            extend: bool                                 = False,
            enforce_order: bool                          = True
        ) -> tuple[NDArray[np.void], float, bool, bool, dict[str, Any]]:

        self.current_task = 0
        if actions is not None:
            if extend:
                self.scheduled_action = np.concatenate([self.scheduled_action, actions])
            
            else:
                self.scheduled_action = actions.copy()

        previous_objective = self.objective.get_current()

        self.n_queries += 1

        stop       = False
        terminated = self.is_terminal()
        truncated  = self.is_truncated()
        time_limit = self.current_time + time_skip if time_skip is not None else MAX_INT

        while not (stop or terminated or truncated):
            stop       = self.one_action(enforce_order, time_limit)
            terminated = self.is_terminal()
            truncated  = self.is_truncated()

        obs    = self._get_obs()
        reward = (self.objective.get_current() - previous_objective) * (-1 if self.minimize else 1)
        info   = self._get_info()

        self.scheduled_action = self.scheduled_action[self.current_task:]

        return obs, reward, terminated, truncated, info



    def one_action(self, enforce_order: bool, time_limit: int) -> bool:
        if self.current_task >= len(self.scheduled_action):
            executing_tasks = self.tasks.is_executing(self.current_time)
            if not np.any(executing_tasks):
                self.current_time = time_limit

                return True

            self.current_time = np.min(self.tasks.end_lb[executing_tasks], initial=time_limit).item()

            return True


        task: Scalar

        available_mask = self.tasks.is_available(self.current_time)
        if enforce_order:
            task = self.scheduled_action[self.current_task]
            task_id = self.tasks.get_indices(task)

            if not available_mask[task_id]:
                executing_tasks = self.tasks.is_executing(self.current_time)

                if not np.any(executing_tasks) or self.current_time >= time_limit:
                    return True

                self.current_time = np.min(self.tasks.end_lb[executing_tasks], initial=time_limit).item()

                return False


        else: # not enforce_order
            task_ids = self.tasks.get_indices(self.scheduled_action[self.current_task:])
            available_schedule = available_mask[task_ids]

            if not np.any(available_schedule):
                to_schedule = self.tasks.is_awaiting()

                self.current_time = np.min(self.tasks.start_lb[to_schedule], initial=time_limit).item()

                return self.current_time >= time_limit


            task_place = self.current_task + int(np.nonzero(available_schedule)[0][0])
            task       = self.scheduled_action[task_place]

            if task_place != self.current_task:
                self.scheduled_action[self.current_task+1:task_place+1] = self.scheduled_action[self.current_task:task_place]
                self.scheduled_action[self.current_task] = task


        self.tasks.fix_start(task, self.current_time)
        self.current_task += 1

        self.update_state()

        to_schedule = self.tasks.is_awaiting()

        if np.any(to_schedule):
            self.current_time = np.min(self.tasks.start_lb[to_schedule], initial=time_limit).item()

        else:
            self.current_time = int(np.max(self.tasks.end_lb[:]))

        return False


    def render_gantt(
            self,
            bin_feature: Optional[str] = None,
            group_feature: Optional[str] = None,
            palette: str = 'cubehelix',
            bar_width: float = 0.8
        ) -> None:
        """
            Render a Gantt chart of the scheduled operations.

        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 6))

        is_fixed = self.tasks.is_fixed()

        start_times = self.tasks.start_lb[is_fixed]
        durations   = self.tasks.durations[is_fixed]

        if group_feature is not None:
            groups = self.tasks[group_feature]
            group_mapping = {group: i for i, group in enumerate(np.unique(groups))}

            groups    = groups[is_fixed]
            group_ids = np.array([group_mapping[group] for group in groups])
            n_groups  = len(group_mapping)

        else:
            group_mapping = {0: 0}

            groups    = np.zeros(len(start_times), dtype=np.int32)
            group_ids = groups
            n_groups  = 1


        if bin_feature is not None:
            bins = self.tasks[bin_feature]

            n_bins = len(np.unique(bins))
            bins   = bins[is_fixed]

        else:
            n_bins = 1
            bins   = np.zeros(len(start_times), dtype=np.int32)

        bins   = self.tasks[bin_feature][is_fixed] if bin_feature is not None else np.zeros(len(start_times), dtype=np.int32)
        n_bins = len(np.unique(self.tasks[bin_feature])) if bin_feature is not None else 1

        colors = np.array(sns.color_palette(palette, n_colors=n_groups))

        ax.barh(
            y         = bins,
            width     = durations,
            left      = start_times,
            color     = colors[group_ids],
            edgecolor = 'white',
            linewidth = 0.5,
            height    = bar_width
        )

        ax.set_yticks(np.arange(n_bins))
        ax.set_ylim(n_bins - bar_width/2, bar_width/2 - 1)
        ax.set_ylabel(bin_feature if bin_feature is not None else 'Task bins')

        ax.set_xlim(0, max(self.current_time/0.95, 1))
        ax.set_xlabel('Time')

        ax.set_title('Gantt Chart of Scheduled Tasks')

        legend_elements = [Patch(facecolor=colors[i], label=group) for i, group in enumerate(group_mapping)]
        ax.legend(handles=legend_elements ,title=group_feature, loc='center left', bbox_to_anchor=(1, 0.5))

        ax.grid(True, alpha=0.4)
        ax.set_axisbelow(True)

        plt.show()


    def render(
            self,
        ) -> None:
        ...
from typing import Any, Optional, Final, Never, Iterable, ClassVar
from numpy.typing import NDArray
from pandas import DataFrame

from mypy_extensions import mypyc_attr

import numpy as np
import numpy.lib.recfunctions as rf

from .constraints import Constraint
from .objectives import Objective
from .variables import IntervalVars, Scalar


MIN_INT: Final[int] = -2 ** 31 + 1
MAX_INT: Final[int] =  2 ** 31 - 1

@mypyc_attr(allow_interpreted_subclasses=True)
class SchedulingCPEnv:
    render_mode: ClassVar[list[str]] = ['gantt']

    constraints : dict[str, Constraint]
    objectives  : dict[str, Objective]

    current_time    : int
    n_queries       : int
    scheduled_action: NDArray[np.generic]
    current_task    : int

    tasks: IntervalVars


    def __init__(
            self,
            instance: DataFrame,
            duration_feature: str | int,
            dataframe_obs: bool = False
        ) -> None:
        self.constraints = {}
        self.objectives  = {}

        tasks = self.process_dataframe(instance)

        self.tasks = IntervalVars(tasks, tasks[duration_feature], instance.index.to_numpy())

        self.dataframe_obs = dataframe_obs


    def process_dataframe(self, df: DataFrame, ignore_index: bool = False) -> NDArray[np.void]:
        if ignore_index:
            df.set_index(
                np.arange(len(self.tasks), len(self.tasks) + len(df))
            )

        return np.asarray(df.to_records(index=False))


    def add_constraint(self, constraint: Constraint, name: Optional[str] = None) -> None:
        if name is None: name = constraint.__class__.__name__

        self.constraints[name] = constraint


    def add_objective(self, objective: Objective, name: Optional[str] = None) -> None:
        if name is None: name = objective.__class__.__name__

        self.objectives[name] = objective


    def export_model(self) -> str:
        model = [
            self.tasks.export_variables(),
            *[constraint.export_constraint() for constraint in self.constraints.values()],
            *[objective.export_objective() for objective in self.objectives.values()]
        ]

        return '\n'.join(model)


    def get_cp_solution(
            self,
            timelimit: float = 60
        ) -> tuple[NDArray[np.generic], NDArray[np.int32], bool]:
        from docplex.cp.model import CpoModel, CpoSolveResult

        model = CpoModel()

        model.set_parameters(TimeLimit=timelimit)
        model.import_model_string(self.export_model())

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

        free_tasks = order[~self.tasks.is_fixed()[order]]

        task_order = self.tasks.ids[free_tasks]

        return task_order, objective_values, is_optimal


    def is_terminal(self) -> bool:
        return bool(self.tasks.is_fixed().all() and (self.current_time >= self.tasks.end_lb[:]).all())


    def is_truncated(self) -> bool:
        return False


    def _get_obs(self) -> NDArray[np.void] | DataFrame:
        obs = self.tasks.get_state(self.current_time)

        if self.dataframe_obs:
            df = DataFrame(obs, index=self.tasks.ids)
            df.index = df.index.rename("task_id")

            return df

        indexed_obs = rf.append_fields(
            obs,
            'task_id',
            self.tasks.ids,
            usemask=False
        )

        fields = obs.dtype.names

        if fields is None:
            return rf.unstructured_to_structured(self.tasks.ids)

        indexed_obs = indexed_obs[['task_id', *fields]]

        return np.asarray(indexed_obs)


    def _get_info(self) -> dict[str, Any]:
        info = {
            'n_queries'         : self.n_queries,
            'current_time'      : self.current_time,
            'executed_actions'  : self.scheduled_action[:self.current_task],
            'scheduled_actions' : self.scheduled_action[self.current_task:],
            'objectives'        : self.get_objective_values()
        }

        if not self.is_terminal(): return info
        
        info |= {
            'solution' : np.argsort(self.tasks.start_lb[:], stable=True).astype(np.int32),
        }

        return info


    def reset(self) -> tuple[NDArray[np.void] | DataFrame, dict[str, Any]]:
        self.current_time = 0
        self.current_task = 0
        self.scheduled_action = np.array([], dtype=self.tasks.index_dtype)

        self.n_queries = 0

        self.tasks.reset_state()
        
        for constraint in self.constraints.values():
            constraint.reset()

        return self._get_obs(), self._get_info()


    # TODO: Implement this method
    def check_env(self) -> None | Never:
        """
            Check if the constraints and objectives are consistent with the tasks.
        """
        raise NotImplementedError()


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


    def get_objective_values(self) -> NDArray[np.float32]:
        return np.array([objective.get_current() for objective in self.objectives.values()], dtype=np.float32)


    def available_actions(self) -> NDArray[np.generic]:
        is_fixed = self.tasks.is_fixed()
        mask = np.less_equal(self.tasks.start_lb[:], self.current_time) & ~is_fixed

        available: NDArray[np.generic] = self.tasks.ids[mask]

        return available


    def step(
            self,
            actions: Optional[NDArray[np.generic]]       = None,
            time_skip: Optional[int]                     = None,
            extend: bool                                 = False,
            enforce_order: bool                          = True
        ) -> tuple[NDArray[np.void] | DataFrame, NDArray[np.float32], bool, bool, dict[str, Any]]:

        self.current_task = 0
        if actions is not None:
            if extend:
                self.scheduled_action = np.concatenate([self.scheduled_action, actions])
            
            else:
                self.scheduled_action = actions.copy()

        if time_skip == 0:
            obs        = self._get_obs()
            reward     = np.zeros(len(self.objectives), dtype=np.float32)
            terminated = self.is_terminal()
            truncated  = self.is_truncated()
            info       = self._get_info()

            return obs, reward, terminated, truncated, info


        previous_objectives = self.get_objective_values()

        self.n_queries += 1

        stop       = False
        terminated = self.is_terminal()
        truncated  = self.is_truncated()
        time_limit = self.current_time + time_skip if time_skip is not None else MAX_INT

        while not (stop or terminated or truncated):
            stop       = self.one_action(enforce_order, time_limit)
            terminated = self.is_terminal()
            truncated  = self.is_truncated()

        obs        = self._get_obs()
        reward     = previous_objectives - self.get_objective_values()
        info       = self._get_info()

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

        available_actions = self.available_actions()
        if enforce_order:
            task = self.scheduled_action[self.current_task]

            if task not in available_actions:
                executing_tasks = self.tasks.is_executing(self.current_time)

                if not np.any(executing_tasks) or self.current_time >= time_limit:
                    return True

                self.current_time = np.min(self.tasks.end_lb[executing_tasks], initial=time_limit).item()

                return False


        else: # not enforce_order
            for task_place, task in enumerate(self.scheduled_action[self.current_task:]):
                if task in available_actions:
                    current     = self.current_task
                    task_place += current

                    if task_place == current: break

                    self.scheduled_action[current+1:task_place+1] = self.scheduled_action[current:task_place]
                    self.scheduled_action[current] = task
                    break


            else:
                to_schedule = self.tasks.is_awaiting()

                self.current_time = np.min(self.tasks.start_lb[to_schedule], initial=time_limit).item()

                return self.current_time >= time_limit


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
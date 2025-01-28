from typing import Any, Optional, Never, ClassVar, TypeVar, TypeAlias, Callable, Concatenate, SupportsFloat, Iterable, SupportsInt, ParamSpec
from pandas import DataFrame

from warnings import warn
import heapq

from .constraints import Constraint
from .objectives import Objective
from .variables import IntervalVars
from .utils import MAX_INT, AVAILABLE_SOLVERS, convert_to_list

from mypy_extensions import mypyc_attr


_T = TypeVar('_T', covariant=True)
_P = ParamSpec('_P')
SetupClass: TypeAlias = Callable[Concatenate[IntervalVars, _P], _T]

def solve_cplex(
        model_string: str,
        var_names: list[str],
        timelimit: Optional[float] = None
    ) -> tuple[list[int], float, bool]:
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

        start_times = [
            int(result.get_var_solution(name).start) for name in var_names # type: ignore
        ]

        model.get_all_variables()

        objective_values = float(result.get_objective_value()) # type: ignore
        is_optimal       = bool(result.is_solution_optimal())

        return start_times, objective_values, is_optimal


def solve_ortools(
        model_string: str,
        var_names: list[str],
        timelimit: Optional[float] = None
    ) -> tuple[list[int], float, bool]:
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

    start_times = [
        cp_solver.Value(locals_[f"{name}_start"]) for name in var_names
    ]

    objective_values = cp_solver.ObjectiveValue()
    is_optimal = status is cp_model.OPTIMAL # type: ignore[comparison-overlap]

    return start_times, objective_values, is_optimal

_Constraint = TypeVar('_Constraint', bound=Constraint)
_Objective = TypeVar('_Objective', bound=Objective)
_Params = ParamSpec('_Params')
@mypyc_attr(allow_interpreted_subclasses=True)
class SchedulingCPEnv:
    metadata: ClassVar[dict[str, Any]] = {
        'render_modes': ['gantt']
    }

    constraints : dict[str, Constraint]
    objective   : Objective
    minimize    : bool
    
    tasks: IntervalVars
    current_time: int
    n_queries: int
    scheduled_actions: list[int]
    current_task: int
    action_sequence: list[int]

    def __init__(
            self,
            instance: DataFrame,
            duration: str | Iterable[int],
        ) -> None:
        self.constraints = {}
        self.objective   = Objective()
        self.minimize    = True

        tasks = {
            feature: instance[feature].tolist() for feature in instance.columns
        }

        if isinstance(duration, str):
            duration_list: list[int] = tasks[duration]
        
        else:
            duration_list = convert_to_list(duration, dtype=int)

        self.tasks = IntervalVars(tasks, duration_list)
        self.scheduled_actions = []
        self.action_sequence   = []


    def add_constraint(
            self,
            constraint_fn: SetupClass[_Params, _Constraint],
            *args: Any, **kwargs: Any
        ) -> None:
        constraint = constraint_fn(self.tasks, *args, **kwargs)

        name = constraint.__class__.__name__
        self.constraints[name] = constraint


    def set_objective(
            self,
            objective_fn: SetupClass[_Params, _Objective],
            minimize: bool = True,
            *args: Any, **kwargs: Any
        ) -> None:
        objective = objective_fn(self.tasks, *args, **kwargs)

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
        ) -> tuple[list[int], list[int], SupportsFloat, bool]:
        model_string = self.export_model(solver)
        var_names    = self.tasks.get_var_name()

        if solver == 'cplex':
            start_times, objective_values, is_optimal = solve_cplex(model_string, var_names, timelimit)

        elif solver == 'ortools':
            start_times, objective_values, is_optimal = solve_ortools(model_string, var_names, timelimit)

        else:
            raise ValueError(f"Unknown solver {solver}, available solvers are 'cplex' or 'ortools'")

        is_fixed = self.tasks.is_fixed()
        arg_order = sorted([
            (start, task) for task, start in enumerate(start_times) if not is_fixed[task]
        ])

        task_order = [task for _, task in arg_order]

        return task_order, start_times, objective_values, is_optimal


    def _get_obs(self) -> dict[str, list[Any]]:
        return self.tasks.get_state(self.current_time)


    def _get_reward(self, previous_objective: float) -> float:
        return (self.objective.get_current(self.current_time) - previous_objective) * (-1 if self.minimize else 1)


    def is_terminal(self) -> bool:
        return all([self.tasks.is_finished(task, self.current_time) for task in range(len(self.tasks))])


    def is_truncated(self) -> bool:
        return False


    def _get_info(self) -> dict[str, Any]:
        return {
            'n_queries'         : self.n_queries,
            'current_time'      : self.current_time,
            'executed_actions'  : self.action_sequence,
            'scheduled_actions' : self.scheduled_actions[self.current_task:],
            'objective_value'   : self.objective.get_current(self.current_time)
        }


    def reset(self) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        self.check_env()

        self.current_time = 0
        self.current_task = 0
        self.scheduled_actions.clear()
        self.action_sequence.clear()

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
            warn("No objective has been set, the environment will provide null rewards.")

        return None


    def get_awaiting_tasks(self) -> list[int]:
        return [task for task in range(len(self.tasks)) if self.tasks.is_awaiting(task)]


    def get_executing_tasks(self) -> list[int]:
        return [task for task in range(len(self.tasks)) if self.tasks.is_executing(task, self.current_time)]


    def get_available_tasks(self) -> list[int]:
        return [task for task in range(len(self.tasks)) if self.tasks.is_available(task, self.current_time)]


    def get_finished_tasks(self) -> list[int]:
        return [task for task in range(len(self.tasks)) if self.tasks.is_finished(task, self.current_time)]


    def get_fixed_tasks(self) -> list[int]:
        return [task for task in range(len(self.tasks)) if self.tasks.is_fixed(task)]


    def advance_to(
            self,
            time: int,
            time_limit: int
        ) -> None:
        self.current_time = max(self.current_time, min(time, time_limit))


    def next_decision_point(
            self,
            time_limit: int
        ) -> None:
        awaiting_tasks = self.get_awaiting_tasks()

        if awaiting_tasks:
            next_time = min(self.tasks.get_start_lb(awaiting_tasks))

        else:
            next_time = max(self.tasks.get_end_lb())

        self.advance_to(next_time, time_limit)


    def advance_to_next_time(
            self,
            time_limit: int
        ) -> None:
        awaiting_tasks = self.get_awaiting_tasks()

        if awaiting_tasks:
            next_time = min([
                self.tasks.get_start_lb(task) for task in awaiting_tasks
                if self.tasks.get_start_lb(task) > self.current_time
            ], default=self.current_time)

        else:
            next_time = max(self.tasks.get_end_lb())

        self.advance_to(next_time, time_limit)


    def update_state(self) -> None:
        for constraint in self.constraints.values():
            constraint.propagate(self.current_time)

    
    def step(
            self,
            action: Optional[Iterable[SupportsInt] | SupportsInt] = None,
            time_skip: Optional[int]    = None,
            extend: bool                = False,
            enforce_order: bool         = True
        ) -> tuple[dict[str, list[Any]], float, bool, bool, dict[str, Any]]:
        if action is not None:
            if not extend:
                self.scheduled_actions.clear()

            next_actions = convert_to_list(action, int)

            self.scheduled_actions.extend(next_actions)

        previous_objective = self.objective.get_current(self.current_time)

        self.current_task = 0
        self.n_queries += 1

        stop       = False
        terminated = self.is_terminal()
        truncated  = self.is_truncated()
        time_limit = self.current_time + time_skip if time_skip is not None else MAX_INT

        while not (stop or terminated or truncated):
            stop = self.one_action(enforce_order, time_limit)

            self.update_state()

            terminated = self.is_terminal()
            truncated  = self.is_truncated()


        obs    = self._get_obs()
        reward = self._get_reward(previous_objective)
        info   = self._get_info()

        self.action_sequence.extend(self.scheduled_actions[:self.current_task])
        self.scheduled_actions = self.scheduled_actions[self.current_task:]

        return obs, reward, terminated, truncated, info


    def one_action(self, enforce_order: bool, time_limit: int) -> bool:
        if self.current_task >= len(self.scheduled_actions):
            self.advance_to_next_time(time_limit)

            return True

        if enforce_order:
            task = self.scheduled_actions[self.current_task]

            if not self.tasks.is_available(task, self.current_time):
                if not self.get_executing_tasks() or self.current_time == time_limit:
                    return True

                self.advance_to_next_time(time_limit)

                return False

        else: # not enforce_order
            for pos, task in enumerate(self.scheduled_actions[self.current_task:]):
                if self.tasks.is_available(task, self.current_time):
                    break

            else: # no available task
                if not self.get_executing_tasks() or self.current_time == time_limit:
                    return True

                self.advance_to_next_time(time_limit)

                return False

            if pos != 0:
                task_place = self.current_task + pos

                self.scheduled_actions[self.current_task+1:task_place+1] = self.scheduled_actions[self.current_task:task_place]
                self.scheduled_actions[self.current_task] = task

        self.tasks.fix_start(task, self.current_time)
        self.current_task += 1

        self.next_decision_point(time_limit)

        return self.current_task >= len(self.scheduled_actions)


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

        fixed_tasks = self.get_fixed_tasks()

        start_times = self.tasks.get_start_lb(fixed_tasks)
        durations   = self.tasks.get_duration(fixed_tasks)

        features    = self.tasks.get_features(fixed_tasks)

        group_mapping: dict[Any, int]
        if group_feature is None:
            group_mapping = {0: 0}
            groups = [0] * len(start_times)

        else:
            group_mapping = {}
            groups = features[group_feature]

            for group in features[group_feature]:
                if group not in group_mapping:
                    group_mapping[group] = len(group_mapping)

        color_pallete = sns.color_palette(palette, n_colors=len(group_mapping))

        if bin_feature is None:
            bins = allocate_bins(self.tasks)

        else:
            bins   = features[bin_feature]

        colors = [color_pallete[group_mapping[group]] for group in groups]

        ax.barh(
            y         = bins,
            width     = durations,
            left      = start_times,
            color     = colors,
            edgecolor = 'white',
            linewidth = 0.5,
            height    = bar_width
        )

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

def allocate_bins(tasks: IntervalVars) -> list[int]:
    fixed_tasks = [task for task in range(len(tasks)) if tasks.is_fixed(task)]

    start_times = tasks.get_start_lb(fixed_tasks)
    end_times   = tasks.get_end_lb(fixed_tasks)

    indexed_tasks = sorted([
        (start_times[i], end_times[i], i) for i in range(len(fixed_tasks))
    ])

    first_start, first_end, first_task = indexed_tasks[0]

    heap = [(first_end, 0)]

    bins = [0] * len(indexed_tasks)

    current_bins = 1
    for start, end, task_id in indexed_tasks[1:]:
        earliest_end, bin_id = heap[0]

        if start >= earliest_end:
            heapq.heappop(heap)
            bin_ = bin_id

        else:
            bin_ = current_bins
            current_bins += 1
        
        bins[task_id] = bin_
        heapq.heappush(heap, (end, bin_))
    
    return bins
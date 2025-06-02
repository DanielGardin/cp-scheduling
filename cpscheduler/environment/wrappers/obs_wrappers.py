from typing import Any, TypeVar, SupportsFloat, Iterable, SupportsInt, Optional

from torch import Tensor, as_tensor
import numpy as np

from ..protocols import WrappedEnv, Env
from ..env import SchedulingCPEnv
from ..utils import convert_to_list, is_iterable_type, AVAILABLE_SOLVERS

_Act = TypeVar('_Act')
_Env = TypeVar('_Env', bound=Env[Any, Any])
class PytorchWrapper(WrappedEnv[Tensor, _Act, _Env]):
    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        obs, info = self.env.reset()

        return as_tensor(obs), info

    def step(self, action: _Act, *args: Any, **kwargs: Any) -> tuple[Tensor, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action, *args, **kwargs)

        return as_tensor(obs), reward, terminated, truncated, info


class End2EndStateWrapper(WrappedEnv[Tensor, SupportsInt | Iterable[SupportsInt], SchedulingCPEnv]):
    job_ops: list[list[int]]

    def __init__(
            self,
            env: SchedulingCPEnv,
            n_future_tasks: int = 3
        ):
        super().__init__(env)
        self.n_future_tasks = n_future_tasks
    
    def process_instance(self, obs: dict[str, list[Any]]) -> None:
        obs = self.env._get_obs()
        self.jobs = obs['job']

        self.n_jobs      = len(set(self.jobs))
        self.job_ops     = [[] for _ in range(self.n_jobs)]
        self.current_ops = [ 0 for _ in range(self.n_jobs)]

        for i in range(self.n_jobs):
            self.current_ops[i] = 0

        # We suppose that the tasks are ordered by operation
        task: int
        for task in obs['task_id']:
            operation: int = obs['operation'][task]
            job: int       = obs['job'][task]

            if operation >= len(self.job_ops[job]):
                self.job_ops[job].extend([-1 for _ in range(operation - len(self.job_ops[job]) + 1)])

            self.job_ops[job][operation] = task

            if obs['buffer'][task] == 'finished' or obs['buffer'][task] == 'executing':
                self.current_ops[job] = operation + 1


    def build_state(self, obs: dict[str, list[Any]]) -> Tensor:
        array_obs = np.array(
            list(zip(*obs.values())),
            dtype=[
                ('task_id', np.int32),
                ('job', np.int32),
                ('operation', np.int32),
                ('machine', np.int32),
                ('processing_time', np.int32),
                ('remaining_time', np.int32),
                ('buffer', 'U9')
            ]
        )

        jobs = np.unique(array_obs['job'])

        is_fixed    = np.array(self.env.tasks.is_fixed())
        lower_bound = np.array(self.env.tasks.get_start_lb())

        order = np.argsort(array_obs, order=['job', 'operation'])

        array_obs   = array_obs[order]
        is_fixed    = is_fixed[order]
        lower_bound = lower_bound[order]

        state      = np.zeros((len(jobs), 2+self.n_future_tasks, 6), dtype=np.int32)

        for i, job in enumerate(jobs):
            job_obs = array_obs[array_obs['job'] == job]

            last_finished = np.max(job_obs['operation'][job_obs['buffer'] == 'finished'], initial=-1)
            n_jobs = len(job_obs)

            job_ops = [
                op if (0 <= op < n_jobs) else -1 for op in range(last_finished, last_finished + self.n_future_tasks + 2)
            ]

            job_mask = np.array(job_ops) != -1

            # Kinda smelly code
            state[i, job_mask, 0] = is_fixed[obs['job'] == job][job_ops][job_mask]
            state[i, job_mask, 1] = lower_bound[obs['job'] == job][job_ops][job_mask]
            state[i, job_mask, 2] = job_obs[job_ops]['processing_time'][job_mask]
            state[i, job_mask, 3] = (job_obs[job_ops]['buffer'] == 'available')[job_mask]
            state[i, job_mask, 4] = job_obs[job_ops]['machine'][job_mask]
            state[i, :, 5]        = job_ops

        return as_tensor(state)
    
    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        obs, info = self.env.reset()

        self.process_instance(obs)

        return self.build_state(obs), info
    
    def step(
            self,
            action: SupportsInt | Iterable[SupportsInt],
            *args: Any, **kwargs: Any
        ) -> tuple[Tensor, SupportsFloat, bool, bool, dict[str, Any]]:
        parsed_action: Optional[list[int]]
        if is_iterable_type(action, SupportsInt):
            parsed_action = convert_to_list(action, int)

        else:
            assert isinstance(action, SupportsInt)
            parsed_action = [int(action)] if int(action) < self.n_jobs else None

        processed_action: Optional[list[int]]
        if parsed_action is None:
            processed_action = None

        else:
            processed_action = []
            for job_idx in parsed_action:
                if job_idx >= self.n_jobs:
                    continue

                idx = self.current_ops[job_idx]

                processed_action.append(self.job_ops[job_idx][idx])

                self.current_ops[job_idx] += 1

        obs, reward, terminated, truncated, info = self.env.step(processed_action, *args, **kwargs)

        return self.build_state(obs), reward, terminated, truncated, info

    def get_cp_solution(
            self,
            timelimit: Optional[float] = None,
            solver: AVAILABLE_SOLVERS = 'cplex'
        ) -> tuple[list[int], list[list[int]], SupportsFloat, bool]:

        task_order, start_times, objective_values, is_optimal = self.env.get_cp_solution(timelimit, solver)

        action = [self.jobs[idx] for idx in task_order]

        parsed_starts = [
            [start_times[task] for task in self.job_ops[job]] for job in range(self.n_jobs)
        ]

        return action, parsed_starts, objective_values, is_optimal
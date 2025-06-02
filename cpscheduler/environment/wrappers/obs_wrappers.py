from typing import Any, TypeVar, SupportsFloat, Iterable, SupportsInt

from torch import Tensor, as_tensor
import numpy as np

from ..protocols import WrappedEnv, Env
from ..env import SchedulingCPEnv

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
    def __init__(
            self,
            env: SchedulingCPEnv,
            n_future_tasks: int = 3
        ):
        super().__init__(env)
        self.n_future_tasks = n_future_tasks
    

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

        return self.build_state(obs), info
    
    def step(
            self,
            action: SupportsInt | Iterable[SupportsInt],
            *args: Any, **kwargs: Any
        ) -> tuple[Tensor, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action, *args, **kwargs)

        return self.build_state(obs), reward, terminated, truncated, info
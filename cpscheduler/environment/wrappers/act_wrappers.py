from typing import Any, SupportsFloat, SupportsInt, Iterable, Optional


from ..env import SchedulingCPEnv
from ..utils import convert_to_list, is_iterable_type, AVAILABLE_SOLVERS
from ..protocols import WrappedEnv

class JobActionWrapper(WrappedEnv[dict[str, list[Any]], SupportsInt | Iterable[SupportsInt], SchedulingCPEnv]):
    job_ops: list[list[int]]
    jobs: list[int]

    def __init__(
            self,
            env: SchedulingCPEnv
        ):
        super().__init__(env)

        self.env = env


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


    def reset(self) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        obs, info = self.env.reset()

        self.process_instance(obs)

        return obs, info


    def step(
            self,
            action: SupportsInt | Iterable[SupportsInt],
            *args: Any, **kwargs: Any
        ) -> tuple[dict[str, list[Any]], SupportsFloat, bool, bool, dict[str, Any]]:
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
        
        return self.env.step(processed_action, *args, **kwargs)


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
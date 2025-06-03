from typing import Any, TypeVar, Iterable

from gymnasium.spaces import Dict, Tuple, Box, OneOf, Space, Sequence

from gymnasium import ObservationWrapper, Env

from ..env import ObsType, ActionType
from ..tasks import Tasks
from ..utils import is_iterable_type
from ..common import MAX_INT

def reshape_space(space: Space[Any], shape: tuple[int, ...]) -> Space[Any]:
    """
    Reshape the space to the given shape.
    """
    if isinstance(space, Box):
        return Box(
            low=space.low.reshape(shape),   # type: ignore
            high=space.high.reshape(shape), # type: ignore
            dtype=space.dtype               # type: ignore
        )

    if isinstance(space, Dict):
        return Dict({key: reshape_space(value, shape) for key, value in space.spaces.items()})

    if isinstance(space, Tuple):
        return Tuple([reshape_space(value, shape) for value in space.spaces])

    if isinstance(space, OneOf):
        return OneOf([reshape_space(value, shape) for value in space.spaces])

    raise ValueError(f"Unsupported space type: {type(space)}")

_Act = TypeVar("_Act")
class TabularObservationWrapper(ObservationWrapper[dict[str, list[Any]], _Act, ObsType]):
    """
    A wrapper that converts the observation space to a single tabular space after merging
    the task and job features.
    """
    def __init__(self, env: Env[ObsType, _Act]):
        super().__init__(env)

        if not env.get_wrapper_attr("loaded"):
            raise ValueError("Environment must be loaded before wrapping.")

        if not is_iterable_type(env.observation_space, Dict):
            raise ValueError(f"Unexpected env observation space: {env.observation_space}")

        task_feature_space, job_feature_space = env.observation_space
        n_tasks = len(env.get_wrapper_attr("tasks"))

        job_spaces = {
            job_feature: reshape_space(space, (n_tasks,))
            for job_feature, space in job_feature_space.spaces.items()
        }
        del job_spaces['job_id']

        self.observation_space = (
            task_feature_space if len(job_feature_space) == 1 else
            Dict({
                **task_feature_space,
                **job_spaces
            })
        )

    def observation(self, observation: ObsType) -> dict[str, list[Any]]:
        task_data, job_data = observation

        if len(job_data) == 1:
            return task_data

        merged_data = task_data.copy()

        jobs_ids: list[int] = task_data['job_id']
        for job_feature in job_data:
            if job_feature == 'job_id':
                continue

            merged_data[job_feature] = [
                job_data[job_feature][job_id]
                for job_id in jobs_ids
            ]

        return merged_data

class CPStateWrapper(ObservationWrapper[ObsType, ActionType, ObsType]):
    """
    A wrapper that adds Constraint Programming (CP) state information
    to the observation space.
    """
    def __init__(self, env: Env[ObsType, ActionType]):
        super().__init__(env)

        if not env.get_wrapper_attr("loaded"):
            raise ValueError("Environment must be loaded before wrapping.")
        
        if not is_iterable_type(env.observation_space, Dict):
            raise ValueError(f"Unexpected env observation space: {env.observation_space}")

        task_feature_space, job_feature_space = env.observation_space
        n_tasks = len(env.get_wrapper_attr("tasks"))

        self.observation_space = Tuple((
            Dict({
                **task_feature_space,
                'lower_bound': Box(low=0, high=MAX_INT, shape=(n_tasks,)),
            }),
            job_feature_space
        ))

    def observation(self, observation: ObsType) -> ObsType:
        task_data, job_data = observation

        tasks: Tasks = self.env.get_wrapper_attr("tasks")

        task_data['lower_bound'] = [
            task.get_start_lb() for task in tasks
        ]

        task_data['upper_bound'] = [
            task.get_start_ub() for task in tasks
        ]

        return task_data, job_data
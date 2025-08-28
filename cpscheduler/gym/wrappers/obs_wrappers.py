from typing import Any, TypeVar
from collections.abc import Iterable
from numpy.typing import NDArray

from abc import ABC, abstractmethod

import numpy as np

from gymnasium.spaces import Dict, Tuple, Box

from gymnasium import ObservationWrapper, Env, Space

from cpscheduler.environment._common import ObsType, MAX_INT as MAX_INT_TIME
from cpscheduler.environment.tasks import Tasks
from cpscheduler.utils.typing_utils import is_iterable_type

from cpscheduler.gym.common import Options

MAX_INT = int(MAX_INT_TIME)

S = TypeVar("S", Space[Any], Box, Dict, Tuple)


def reshape_space(space: S, shape: tuple[int, ...]) -> S:
    "Reshape the space to the given shape."
    if isinstance(space, Box):
        return Box(
            low=space.low.reshape(shape),
            high=space.high.reshape(shape),
        )

    if isinstance(space, Dict):
        return Dict(
            {key: reshape_space(value, shape) for key, value in space.spaces.items()}
        )

    if isinstance(space, Tuple):
        return Tuple(reshape_space(value, shape) for value in space.spaces)

    raise ValueError(f"Unsupported space type: {type(space)}")


_Obs = TypeVar("_Obs")
_Act = TypeVar("_Act")


class SchedulingObservationWrapper(ObservationWrapper[_Obs, _Act, ObsType], ABC):
    def __init__(self, env: Env[ObsType, _Act]):
        super().__init__(env)
        self.observation_space = self.get_observation_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Options = None,
    ) -> tuple[_Obs, dict[str, Any]]:
        previously_loaded = self.get_wrapper_attr("loaded")

        obs, info = super().reset(seed=seed, options=dict(options) if options else None)

        if options is not None or not previously_loaded:
            self.observation_space = self.get_observation_space()

        return obs, info

    @property
    def _core_observation_space(self) -> Iterable[Dict]:
        space = self.env.get_wrapper_attr("get_observation_space")()

        if is_iterable_type(space, Dict):
            return space

        raise ValueError(
            f"Unexpected env observation space: {self.env.observation_space}"
        )

    @abstractmethod
    def get_observation_space(self) -> Space[_Obs]:
        """
        Get the observation space for the environment.
        This method is called when the environment is loaded, both during
        initialization and when the environment is reset.
        """


class TabularObservationWrapper(
    SchedulingObservationWrapper[dict[str, list[Any]], _Act]
):
    """
    A wrapper that converts the observation space to a single tabular space after merging
    the task and job features.
    """

    def __init__(self, env: Env[ObsType, _Act]):
        super().__init__(env)

    def get_observation_space(self) -> Space[dict[str, list[Any]]]:
        task_feature_space, job_feature_space = self._core_observation_space
        n_tasks = len(self.env.get_wrapper_attr("tasks"))

        job_spaces = {
            job_feature: reshape_space(space, (n_tasks,))
            for job_feature, space in job_feature_space.spaces.items()
        }
        del job_spaces["job_id"]

        return (
            task_feature_space
            if len(job_feature_space) == 1
            else Dict({**task_feature_space, **job_spaces})
        )

    def observation(self, observation: ObsType) -> dict[str, list[Any]]:
        task_data, job_data = observation

        merged_data = {
            task_feature: task_data[task_feature] for task_feature in task_data
        }

        jobs_ids: list[int] = task_data["job_id"]
        for job_feature in job_data:
            if job_feature == "job_id":
                continue

            merged_data[job_feature] = [
                job_data[job_feature][job_id] for job_id in jobs_ids
            ]

        return merged_data


class ArrayObservationWrapper(SchedulingObservationWrapper[NDArray[np.float32], _Act]):
    """
    A wrapper that converts the observation space to a single array space after merging
    the task and job features.
    """

    def __init__(self, env: Env[ObsType, _Act], features: Iterable[str] | None = None):
        self.features = list(features) if features is not None else []

        super().__init__(env)

    def get_observation_space(self) -> Space[NDArray[np.float32]]:
        task_feature_space, job_feature_space = self._core_observation_space

        n_features = (
            len(self.features)
            if self.features
            else len(task_feature_space)
            + len(job_feature_space)
            - 1  # Exclude 'job_id'
        )

        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(len(self.env.get_wrapper_attr("tasks")), n_features),
            dtype=np.float32,
        )

    def observation(self, observation: ObsType) -> NDArray[np.float32]:
        task_data, job_data = observation
        n_tasks = len(self.env.get_wrapper_attr("tasks"))

        features = (
            self.features
            if self.features
            else [
                *task_data.keys(),
                *(feature for feature in job_data if feature != "job_id"),
            ]
        )
        obs = np.zeros((n_tasks, len(features)), dtype=np.float32)

        for i, feature in enumerate(features):
            if feature in task_data:
                obs[:, i] = task_data[feature]

            elif feature in job_data:
                obs[:, i] = [
                    job_data[feature][job_id] for job_id in task_data["job_id"]
                ]

            else:
                raise ValueError(f"Feature '{feature}' not found in task or job data.")

        return obs


class CPStateWrapper(SchedulingObservationWrapper[ObsType, _Act]):
    """
    A wrapper that adds Constraint Programming (CP) state information
    to the observation space.
    """

    def get_observation_space(self) -> Space[ObsType]:
        task_feature_space, job_feature_space = self._core_observation_space
        n_tasks = len(self.env.get_wrapper_attr("tasks"))

        return Tuple(
            (
                Dict(
                    {
                        **task_feature_space,
                        "lower_bound": Box(low=0, high=MAX_INT, shape=(n_tasks,)),
                        "upper_bound": Box(low=0, high=MAX_INT, shape=(n_tasks,)),
                    }
                ),
                job_feature_space,
            )
        )

    def observation(self, observation: ObsType) -> ObsType:
        task_data, job_data = observation

        tasks: Tasks = self.env.get_wrapper_attr("tasks")

        task_data["lower_bound"] = [task.get_start_lb() for task in tasks]

        task_data["upper_bound"] = [task.get_start_ub() for task in tasks]

        return task_data, job_data

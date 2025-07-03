from typing import Any, TypeVar
from collections.abc import Callable
from numpy.typing import NDArray

from abc import ABC, abstractmethod

import numpy as np
from gymnasium.spaces import Dict, Tuple, Box, OneOf, Space

from gymnasium import ObservationWrapper, Env

from ..tasks import Tasks
from ..utils import is_iterable_type
from .._common import ObsType, MAX_INT as MAX_INT_TIME

MAX_INT = int(MAX_INT_TIME)


def reshape_space(space: Space[Any], shape: tuple[int, ...]) -> Space[Any]:
    """
    Reshape the space to the given shape.
    """
    if isinstance(space, Box):
        return Box(
            low=space.low.reshape(shape),  # type: ignore
            high=space.high.reshape(shape),  # type: ignore
            dtype=space.dtype,  # type: ignore
        )

    if isinstance(space, Dict):
        return Dict(
            {key: reshape_space(value, shape) for key, value in space.spaces.items()}
        )

    if isinstance(space, Tuple):
        return Tuple([reshape_space(value, shape) for value in space.spaces])

    if isinstance(space, OneOf):
        return OneOf([reshape_space(value, shape) for value in space.spaces])

    raise ValueError(f"Unsupported space type: {type(space)}")


_Obs = TypeVar("_Obs")
_Act = TypeVar("_Act")


class SchedulingObservationWrapper(ObservationWrapper[_Obs, _Act, ObsType], ABC):
    def __init__(self, env: Env[ObsType, _Act]):
        super().__init__(env)

        if self.env.get_wrapper_attr("loaded"):
            self.observation_space = self.get_observation_space()

        else:
            default_space = self.default_observation_space()

            if default_space is not None:
                self.observation_space = default_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[_Obs, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.observation_space = self.get_observation_space()

        return obs, info

    @abstractmethod
    def get_observation_space(self) -> Space[_Obs]:
        """
        Get the observation space for the environment.
        This method is called when the environment is loaded, both during
        initialization and when the environment is reset.
        """

    def default_observation_space(self) -> Space[_Obs] | None:
        """
        Get the default observation space for the environment during initialization,
        when the environment's observation space is not known yet.
        """
        return None


class TabularObservationWrapper(
    SchedulingObservationWrapper[dict[str, list[Any]], _Act]
):
    """
    A wrapper that converts the observation space to a single tabular space after merging
    the task and job features.
    """

    def get_observation_space(self) -> Space[dict[str, list[Any]]]:
        if not is_iterable_type(self.env.observation_space, Dict):
            raise ValueError(
                f"Unexpected env observation space: {self.env.observation_space}"
            )

        task_feature_space, job_feature_space = self.env.observation_space
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

        if len(job_data) == 1:
            return task_data

        merged_data = task_data.copy()

        jobs_ids: list[int] = task_data["job_id"]
        for job_feature in job_data:
            if job_feature == "job_id":
                continue

            merged_data[job_feature] = [
                job_data[job_feature][job_id] for job_id in jobs_ids
            ]

        return merged_data


class CPStateWrapper(SchedulingObservationWrapper[ObsType, _Act]):
    """
    A wrapper that adds Constraint Programming (CP) state information
    to the observation space.
    """

    def get_observation_space(self) -> Space[ObsType]:
        if not is_iterable_type(self.env.observation_space, Dict):
            raise ValueError(
                f"Unexpected env observation space: {self.env.observation_space}"
            )

        task_feature_space, job_feature_space = self.env.observation_space
        n_tasks = len(self.env.get_wrapper_attr("tasks"))

        return Tuple(
            (
                Dict(
                    {
                        **task_feature_space,
                        "lower_bound": Box(low=0, high=MAX_INT, shape=(n_tasks,)),
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


class PreprocessObservationWrapper(
    SchedulingObservationWrapper[NDArray[np.floating[Any]], _Act]
):
    """
    A wrapper that preprocesses the observation space by removing the 'job_id' feature
    from the job features.
    """

    def __init__(
        self,
        env: Env[ObsType, _Act],
        transform: Callable[[*ObsType], NDArray[np.floating[Any]]],
    ):
        self.transform = transform

        obs, info = env.reset()
        array_obs = transform(*obs)

        self.n_features = array_obs.shape[-1]

        super().__init__(env)

    def get_observation_space(self) -> Space[NDArray[np.floating[Any]]]:
        if not is_iterable_type(self.env.observation_space, Dict):
            raise ValueError(
                f"Unexpected env observation space: {self.env.observation_space}"
            )

        n_jobs = len(getattr(self.env.get_wrapper_attr("tasks"), "jobs"))

        return Box(float("-inf"), float("inf"), shape=(n_jobs, self.n_features))

    def observation(self, observation: ObsType) -> NDArray[np.floating[Any]]:
        task_data, job_data = observation

        return self.transform(task_data, job_data)

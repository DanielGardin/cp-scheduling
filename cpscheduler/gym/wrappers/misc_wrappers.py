from typing import Any, TypeVar
from collections.abc import Iterable

from abc import ABC

import random

from gymnasium import Env, Wrapper, Space

from cpscheduler.environment._common import (
    InstanceTypes,
    ProcessTimeAllowedTypes,
    InstanceConfig,
)
from cpscheduler.common import unwrap_env

from cpscheduler.gym.common import Options, get_instance_config

_Obs = TypeVar("_Obs")
_Act = TypeVar("_Act")


class InstanceWrapper(Wrapper[_Obs, _Act, _Obs, _Act], ABC):
    """
    A base wrapper class for scheduling environments that allows for instance
    generation and manipulation during initialization and reset.

    Parameters
    ----------
    env : Env
        The scheduling environment to wrap.
    """

    def __init__(
        self,
        env: Env[_Obs, _Act],
        initial_config: Options = None,
        seed: int | None = None,
    ):
        super().__init__(env)

        instance_config = self.parse_options(seed, initial_config)

        if instance_config is not None:
            wrapped_env = unwrap_env(self.env)
            wrapped_env.set_instance(**instance_config)

            self.observation_space = self.get_observation_space()

    def get_observation_space(self) -> Space[_Obs]:
        return self.env.get_wrapper_attr("get_observation_space")()  # type: ignore[no-any-return]

    def parse_options(
        self, seed: int | None, options: Options
    ) -> InstanceConfig | None:
        instance_config: InstanceConfig | None = None
        if not options:
            instance_config = self.generate_instance(seed)

        else:
            instance_config = get_instance_config(options)

        if instance_config is not None:
            instance_config = self.manipulate_instance(instance_config)

        return instance_config

    def generate_instance(self, seed: int | None) -> InstanceConfig:
        """
        Generate a new instance configuration for the environment.

        Parameters
        ----------
        seed : int | None
            An optional seed for random number generation.

        Returns
        -------
        InstanceConfig
            The generated instance configuration.
        """
        return {}

    def manipulate_instance(self, instance_config: InstanceConfig) -> InstanceConfig:
        """
        Manipulate the instance configuration before resetting the environment.

        Parameters
        ----------
        instance_config : InstanceConfig
            The instance configuration to manipulate.

        Returns
        -------
        InstanceConfig
            The manipulated instance configuration.
        """
        return instance_config

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Options = None,
    ) -> tuple[_Obs, dict[str, Any]]:
        previously_loaded = self.get_wrapper_attr("loaded")
        instance_config = self.parse_options(seed, options)

        obs, info = super().reset(
            seed=seed, options=dict(instance_config) if instance_config else None
        )

        if instance_config or not previously_loaded:
            self.observation_space = self.get_observation_space()

        return obs, info


class InstancePoolWrapper(InstanceWrapper[_Obs, _Act]):
    """
    A wrapper that generates a random instance of the scheduling problem from a pool
    of instances every time the environment is reset.

    Parameters
    ----------
    env : Env
        The scheduling environment to wrap.

    instances : list[DataFrame]
        A list of DataFrames representing different scheduling problem instances.
    """

    def __init__(
        self,
        env: Env[_Obs, _Act],
        instance_pool: Iterable[InstanceTypes],
        processing_times: ProcessTimeAllowedTypes | None = None,
        job_instance_pool: Iterable[InstanceTypes] | None = None,
        job_feature: str = "",
    ):
        self.instances = list(instance_pool)
        self.processing_times = processing_times
        self.job_instance_pool = list(job_instance_pool) if job_instance_pool else None
        self.job_feature = job_feature

        self.n_instances = len(self.instances)

        super().__init__(env)

    def generate_instance(self, seed: int | None) -> InstanceConfig:
        rng = random.Random(seed)

        instance_idx = rng.randint(0, self.n_instances - 1)

        return {
            "instance": self.instances[instance_idx],
            "processing_times": self.processing_times,
            "job_instance": (
                self.job_instance_pool[instance_idx]
                if self.job_instance_pool is not None
                else {}
            ),
            "job_feature": self.job_feature,
        }

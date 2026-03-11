from typing import Any, TypeVar
from collections.abc import Iterable

from abc import ABC

from gymnasium import Env, Wrapper, Space

from cpscheduler.utils._protocols import (
    Options,
    InstanceTypes,
    InstanceConfig,
)
from cpscheduler.common import unwrap_env

from cpscheduler.gym.common import get_instance_config

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

        instance_config = self._parse_options(seed, initial_config)

        if instance_config is not None:
            wrapped_env = unwrap_env(self.env)
            wrapped_env.set_instance(**instance_config)

            self.observation_space = self.get_observation_space()

    def get_observation_space(self) -> Space[_Obs]:
        return self.env.get_wrapper_attr("get_observation_space")()  # type: ignore[no-any-return]

    def _parse_options(
        self, seed: int | None, options: Options
    ) -> InstanceConfig | None:
        if not options:
            options = self.generate_instance()

        instance_config = get_instance_config(options)

        if instance_config is not None:
            instance_config = self.manipulate_instance(instance_config)

        return instance_config

    def generate_instance(self) -> InstanceConfig:
        """
        Generate a new instance configuration for the environment.

        Returns
        -------
        InstanceConfig
            The generated instance configuration.
        """
        return {}

    def manipulate_instance(
        self, instance_config: InstanceConfig
    ) -> InstanceConfig:
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
        instance_config = self._parse_options(seed, options)

        obs, info = super().reset(
            seed=seed,
            options=dict(instance_config) if instance_config else None,
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

    instances : Iterable[InstanceTypes]
        An iterable of instances, randomly sampled during environment reset.

    p : list[float] | None
        A list of probabilities associated with each instance in the pool.
    """

    def __init__(
        self,
        env: Env[_Obs, _Act],
        instance_pool: Iterable[InstanceTypes],
        p: list[float] | None = None,
    ):
        self.instances = list(instance_pool)

        if p is not None:
            if len(p) != len(self.instances):
                raise ValueError(
                    "Length of probability list must match number of instances."
                )

            if not abs(sum(p) - 1.0) < 1e-8:
                raise ValueError("Probabilities must sum to 1.")

            if any(prob < 0 for prob in p):
                raise ValueError("Probabilities must be non-negative.")

            self.p = p

        else:
            self.p = [1.0 / len(self.instances)] * len(self.instances)

        super().__init__(env)

    def generate_instance(self) -> InstanceConfig:
        idx = self.np_random.choice(len(self.instances), p=self.p)

        instance = self.instances[idx]

        return {"instance": instance}

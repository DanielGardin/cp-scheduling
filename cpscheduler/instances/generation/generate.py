from typing import Any
from typing_extensions import Unpack

from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment._common import InstanceConfig
from cpscheduler.common import unwrap_env

from ._common import InstanceGeneratorConfig
from .setup import generate_base_instance


MAX_ENV_DEPTH = 10  # Maximum depth for the environment wrapping


class InstanceGenerator:
    """
    Class to automatically generate instances based on the provided environment.

    Attributes:
        env: SchedulingEnv
            The environment containing the scheduling problem.
    """

    def __init__(self, env: Any | SchedulingEnv):
        self.env = unwrap_env(env)

    def generate_instance(
        self, **kwargs: Unpack[InstanceGeneratorConfig]
    ) -> InstanceConfig:
        """
        Generate a scheduling instance based on the environment and provided configurations.

        Parameters
        ----------
        kwargs: Unpack[InstanceGeneratorConfig]
            Additional configurations for instance generation.

        Returns
        -------
        InstanceConfig
            The generated scheduling instance.
        """
        base_instance = generate_base_instance(self.env.setup, kwargs)

        return base_instance

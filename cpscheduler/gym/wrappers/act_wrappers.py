from typing import Any, TypeVar
from collections.abc import Iterable

from abc import ABC, abstractmethod

from gymnasium import ActionWrapper, Env
from gymnasium.spaces import Space, Box, Sequence

from numpy import int64

from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.instructions import ActionType
from cpscheduler.environment._common import Int

from cpscheduler.gym import SchedulingEnvGym

_Obs = TypeVar("_Obs", bound=Any)
_Act = TypeVar("_Act")


class SchedulingActionWrapper(ActionWrapper[_Obs, _Act, ActionType], ABC):
    def __init__(self, env: Env[_Obs, ActionType] | SchedulingEnv):
        if isinstance(env, SchedulingEnv):
            wrapped_env: Env[Any, ActionType] = SchedulingEnvGym.from_env(env)
            super().__init__(wrapped_env)  # type: ignore[call-arg]

        else:
            super().__init__(env)

        if self.env.get_wrapper_attr("loaded"):
            self.action_space = self.get_action_space()

        else:
            default_space = self.default_action_space()

            if default_space is not None:
                self.action_space = default_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[_Obs, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.action_space = self.get_action_space()

        return obs, info

    @abstractmethod
    def get_action_space(self) -> Space[_Act]:
        """
        Get the action space for the environment.
        This method is called when the environment is loaded, both during
        initialization and when the environment is reset.
        """

    def default_action_space(self) -> Space[_Act] | None:
        """
        Get the default action space for the environment during initialization,
        when the environment's action space is not known yet.
        """
        return None


class PermutationActionWrapper(SchedulingActionWrapper[_Obs, Iterable[Int]]):
    """
    A wrapper that converts the action space to a permutation of the job IDs.
    """

    def __init__(self, env: Env[_Obs, ActionType] | SchedulingEnv, strict: bool = True):
        super().__init__(env)

        self.instruction = "execute" if strict else "submit"

    def get_action_space(self) -> Space[Iterable[Int]]:
        n_jobs = len(getattr(self.env.get_wrapper_attr("tasks"), "jobs"))

        return Sequence(
            Box(low=0, high=n_jobs - 1, shape=(1,), dtype=int64), stack=True
        )

    def action(self, action: Iterable[Int]) -> ActionType:
        return [(self.instruction, job_id) for job_id in action]

from typing import Any, TypeVar, SupportsInt
from collections.abc import Iterable

from abc import ABC, abstractmethod

from gymnasium.spaces import Space, Box, Sequence

from gymnasium import ActionWrapper, Env

from cpscheduler.environment.instructions import ActionType

_Obs = TypeVar("_Obs")
_Act = TypeVar("_Act")


class SchedulingActionWrapper(ActionWrapper[_Obs, _Act, ActionType], ABC):
    def __init__(self, env: Env[_Obs, ActionType]):
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


class PermutationActionWrapper(SchedulingActionWrapper[_Obs, Iterable[SupportsInt]]):
    """
    A wrapper that converts the action space to a permutation of the job IDs.
    """

    def __init__(self, env: Env[_Obs, ActionType], strict: bool = True):
        super().__init__(env)

        self.instruction = "execute" if strict else "submit"

    def get_action_space(self) -> Space[Iterable[SupportsInt]]:
        n_jobs = len(getattr(self.env.get_wrapper_attr("tasks"), "jobs"))

        return Sequence(Box(low=0, high=n_jobs - 1, shape=(1,)), stack=True)

    def action(self, action: Iterable[SupportsInt]) -> ActionType:
        return [(self.instruction, job_id) for job_id in action]

from typing import Any, TypeVar
from collections.abc import Iterable

from abc import ABC, abstractmethod

from numpy import int64

from gymnasium import ActionWrapper, Env
from gymnasium.spaces import Space, Box, Sequence

from cpscheduler.environment.instructions import ActionType
from cpscheduler.environment._common import Int, Options

_Obs = TypeVar("_Obs")
_Act = TypeVar("_Act")


class SchedulingActionWrapper(ActionWrapper[_Obs, _Act, ActionType], ABC):
    requires_loaded: bool = False

    def __init__(self, env: Env[_Obs, ActionType]):
        super().__init__(env)

        if not self.requires_loaded or self.env.get_wrapper_attr("loaded"):
            self.action_space = self.get_action_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Options = None,
    ) -> tuple[_Obs, dict[str, Any]]:
        previously_loaded = self.get_wrapper_attr("loaded")

        obs, info = super().reset(seed=seed, options=dict(options) if options else None)

        if self.requires_loaded and (options is not None or not previously_loaded):
            self.action_space = self.get_action_space()

        return obs, info

    @abstractmethod
    def get_action_space(self) -> Space[_Act]:
        """
        Get the action space for the environment.
        This method is called when the environment is loaded, both during
        initialization and when the environment is reset.
        """


# TODO: Make it job-oriented to allow job-shop and flow-shop scheduling
class PermutationActionWrapper(SchedulingActionWrapper[_Obs, Iterable[Int]]):
    """
    A wrapper that converts the action space to a permutation of the job IDs.
    """

    def __init__(
        self,
        env: Env[_Obs, ActionType],
        strict: bool = True,
        job_oriented: bool = False,
    ):
        super().__init__(env)

        self.instruction = "execute" if strict else "submit"

        if job_oriented:
            self.instruction += " job"

        self.job_oriented = job_oriented

    def get_action_space(self) -> Space[Iterable[Int]]:
        return Sequence(Box(low=0, high=2**31 - 1, dtype=int64), stack=True)

    def action(self, action: Iterable[Int]) -> ActionType:
        return [(self.instruction, job_id) for job_id in action]

"""Action wrappers for Gymnasium environments."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal, TypeVar

from gymnasium import ActionWrapper, Env
from gymnasium.spaces import Box, Sequence, Space
from numpy import int64
from typing_extensions import override

from cpscheduler.environment.constants import Int
from cpscheduler.environment.des import ActionType
from cpscheduler.environment.utils.protocols import Options

_Obs = TypeVar("_Obs")
_Act = TypeVar("_Act")


class SchedulingActionWrapper(ActionWrapper[_Obs, _Act, ActionType], ABC):
    """Base class for action wrappers for scheduling environments.

    This wrapper is a specialized version of the gymnasium.ActionWrapper for
    the SchedulingEnvGym environment.

    """

    requires_loaded: bool = False

    def __init__(self, env: Env[_Obs, ActionType]):
        super().__init__(env)

        if not self.requires_loaded or self.env.get_wrapper_attr("loaded"):
            self.action_space = self.get_action_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Options | None = None,
    ) -> tuple[_Obs, dict[str, Any]]:
        """Reset the environment and update the action space if necessary."""
        previously_loaded = self.get_wrapper_attr("loaded")

        obs, info = super().reset(
            seed=seed, options=dict(options) if options else None
        )

        if self.requires_loaded and (
            options is not None or not previously_loaded
        ):
            self.action_space = self.get_action_space()

        return obs, info

    @abstractmethod
    def get_action_space(self) -> Space[_Act]:
        """Get the action space for the environment.

        This method is called when the environment loads a new instance,
        potentially changing the action space.
        """


class PermutationActionWrapper(SchedulingActionWrapper[_Obs, Iterable[Int]]):
    """A wrapper that converts the action space to a permutation of the job IDs."""

    instruction: Literal["execute", "submit"]

    def __init__(
        self,
        env: Env[_Obs, ActionType],
        schedule_generation: Literal["serial", "parallel"] = "serial",
    ):
        """Initialize the PermutationActionWrapper.

        Parameters
        ----------
        env : Env[Any, ActionType]
            The environment to wrap.

        schedule_generation : {"serial", "parallel"}, default="serial"
            The method used to generate schedules in the environment.
            - "serial": The environment executes one job at a time, in the order
                specified by the action.

            - "parallel": The environment submits all jobs at once, the order of
                jobs encodes its priority.
                The environment always chooses the job with the highest priority
                that is currently available to execute.

        """
        super().__init__(env)

        if schedule_generation == "serial":
            self.instruction = "execute"

        elif schedule_generation == "parallel":
            self.instruction = "submit"

        else:
            raise ValueError(
                f"Invalid schedule generation method: {schedule_generation}"
            )

    @override
    def get_action_space(self) -> Space[Iterable[Int]]:
        return Sequence(Box(low=0, high=2**31 - 1, dtype=int64), stack=True)

    @override
    def action(self, action: Iterable[Int]) -> ActionType:
        return ((self.instruction, job_id) for job_id in action)

from typing import Any, TypeVar, Iterable

from gymnasium.spaces import Box, Sequence

from gymnasium import ActionWrapper, Env

from ..env import ActionType

_Obs = TypeVar("_Obs")

class PermutationActionWrapper(ActionWrapper[_Obs, Iterable[int], ActionType]):
    """
    A wrapper that converts the action space to a permutation of the job IDs.
    """
    def __init__(
            self,
            env: Env[_Obs, ActionType],
            strict: bool = True
        ):
        super().__init__(env)

        if not env.get_wrapper_attr("loaded"):
            raise ValueError("Environment must be loaded before wrapping.")

        n_jobs = len(getattr(env.get_wrapper_attr("tasks"), "jobs"))
        self.instruction = "execute" if strict else "submit"

        self.action_space = Sequence(
            Box(low=0, high=n_jobs - 1, shape=(1,)), stack=True
        )

    def action(self, action: Iterable[int]) -> ActionType:
        return [(self.instruction, job_id) for job_id in action]


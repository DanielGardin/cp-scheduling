"""Miscellaneous wrappers for alternative transitions and information."""

from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar

from gymnasium import Env, Wrapper
from typing_extensions import override

from cpscheduler.environment.backend import ActionType
from cpscheduler.gym.common import ActionSpace

if TYPE_CHECKING:
    from cpscheduler.environment import SchedulingEnv


_Obs = TypeVar("_Obs")
_Act = TypeVar("_Act")


class ForcedActionWrapper(Wrapper[_Obs, ActionType, _Obs, ActionType]):
    """Automatically commits forced actions without querying the policy.

    For some backends, there are states where there is only one feasible action,
    making the decision fully determined by the environment. In such states,
    querying the policy provides no useful learning signal because the action
    always has probability one, regardless of the policy parameters.

    This wrapper detects states with a singleton action support and automatically
    commits the only feasible action. The policy is queried only when the state
    contains multiple feasible actions.

    Note
    ----
    This wrapper has an assumption that the chosen backend has an "execute"
    instruction.
    There is another underlying assumption regarding the rewards, forced action
    rewards are accumulated in the initial reward to allow credit assignment
    to the action, as if it provokes the chain of forced actions.
    """

    def __init__(self, env: Env[_Obs, ActionType]):
        super().__init__(env)

        if env.action_space != ActionSpace:
            raise ValueError(
                f"Environment {type(env).__name__} must accept raw "
                "instruction tuples as the action shape. This may be caused by "
                "an action wrapper on top of the core env. To solve this error "
                "apply the ForcedActionWrapper before any action wrapper."
            )

    @override
    def step(
        self, action: ActionType
    ) -> tuple[_Obs, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, term, trunc, info = self.env.step(action)
        reward = float(reward)

        core_env: SchedulingEnv = self.env.get_wrapper_attr("core")
        backend = core_env.backend

        if backend.is_empty():
            support = core_env.get_action_support()

            while len(support) == 1:
                out = self.env.step(("execute", support[0]))
                support = core_env.get_action_support()

                obs, forced_reward, term, trunc, info = out
                reward += float(forced_reward)

        return obs, reward, term, trunc, info

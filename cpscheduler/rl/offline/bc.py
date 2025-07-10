from typing import Any
from torch import Tensor

import torch
from torch.optim import Optimizer

from tensordict import TensorDict

from ..base import BaseAlgorithm
from ..buffer import Buffer
from ..protocols import Policy


class BehaviorCloning(BaseAlgorithm):
    """
    Behavior Cloning algorithm.

    Behavior Cloning is a supervised learning algorithm that learns a policy
    by mimicking the actions of an expert. It uses a dataset of state-action
    pairs to train a neural network to predict the action given a state.

    Parameters:
    ----------
    states: Tensor
        The states collected from the expert.

    actions: Tensor
        The actions taken by the expert.

    actor: Policy
        The actor network that predicts the action given a state.
        It should take the state as input and return the predicted action
        and log probability of the action.

    actor_optimizer: Optimizer
        The optimizer used to update the actor network.
        It should be an instance of torch.optim.Optimizer or a subclass of it.
    """

    def __init__(
        self,
        states: Tensor,
        actions: Tensor,
        actor: Policy[Tensor, Tensor],
        actor_optimizer: Optimizer,
        device: str = "auto",
    ):
        buffer = Buffer.from_tensors(
            state=states,
            action=actions,
        )

        super().__init__(buffer, device)

        self.actor = actor
        self.actor_optimizer = actor_optimizer

    def update(self, batch: TensorDict) -> dict[str, Any]:
        target_action = batch["action"]

        loss = -torch.mean(self.actor.log_prob(batch["state"], target_action))

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "loss/actor": loss.item(),
        }

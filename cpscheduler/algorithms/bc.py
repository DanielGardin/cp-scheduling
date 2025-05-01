from typing import Any, Dict
from torch import Tensor
from torch import nn
from torch.optim import Optimizer

from tensordict import TensorDict

from .base import BaseAlgorithm
from .buffer import Buffer


class BehaviorCloning(BaseAlgorithm):
    def __init__(
            self,
            data: Buffer,
            loss: nn.Module,
            actor: nn.Module,
            actor_optimizer: Optimizer,
        ):
        super().__init__(data)

        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.loss_fn = loss


    def train_step(self, batch: TensorDict) -> dict[str, Any]:
        action, log_prob, _ = self.actor.get_action(batch['state'])

        target_action = batch['action']
        action        = action.view(target_action.size())

        loss = self.loss_fn(action, log_prob, target_action)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "loss/actor" : loss.item()
        }
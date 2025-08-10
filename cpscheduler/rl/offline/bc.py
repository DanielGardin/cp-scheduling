from typing import Any
from torch.types import Tensor, Device

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
        device: Device = "auto",
    ):
        buffer = Buffer.from_tensors(
            state=states,
            action=actions,
        )

        super().__init__(buffer, actor, device)

        self.actor_optimizer = actor_optimizer

    def update(self, batch: TensorDict) -> dict[str, Any]:
        target_action = batch["action"]

        loss = -torch.mean(self.policy.log_prob(batch["state"], target_action))

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "loss/actor": loss.item(),
        }


class EntropyAnnealingBC(BehaviorCloning):
    """
    Behavior Cloning with Entropy Annealing algorithm.

    This class extends the Behavior Cloning algorithm by annealing the policy's 
    entropy over time, encouraging exploration of local minima, and then gradually
    incresing the confidence of the policy to focus on the most promising actions.

    This is achieved by introducing an exponential temperature schedule that decreases
    over time, converging to 0 as the target number of updates is reached.

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

    initial_temperature: float
        The initial temperature for the entropy term.
        Default is 1.0.

    decay_rate: float
        The rate at which the temperature decays.
        Default is 0.99.

    device: Device  
        The device to run the algorithm on.
        Default is "auto".
    """
    def __init__(
        self,
        states: Tensor,
        actions: Tensor,
        actor: Policy[Tensor, Tensor],
        actor_optimizer: Optimizer,
        initial_temperature: float = 1.0,
        decay_rate: float = 0.99,
        device: Device = "auto",
    ):
        super().__init__(states, actions, actor, actor_optimizer, device)

        self.decay_rate = decay_rate
        self.current_temperature = initial_temperature

    def update(self, batch: TensorDict) -> dict[str, Any]:
        target_action = batch["action"]

        loss = -torch.mean(self.policy.log_prob(batch["state"], target_action, temp=self.current_temperature))

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "loss/actor": loss.item(),
        }
    
    def on_epoch_end(self) -> dict[str, Any]:
        self.current_temperature *= self.decay_rate
        return {
            "temperature": self.current_temperature,
        }

from typing import Any, Sequence, Self

import torch
import torch.nn as nn
from torch.optim import Optimizer

from tensordict import TensorDict

from .base import BaseAlgorithm
from .buffer import Buffer
from .utils import turn_off_grad


from copy import deepcopy

class IQL(BaseAlgorithm):
    """
    Implicit Q-Learning (IQL) algorithm.

    Parameters
    ----------
    data : Buffer
        Buffer to store the training data.

    actor : nn.Module
        Actor network that generates actions. The input must be the state and the output
        is a tuple of action and log probability.
    
    critic_ensemble : Sequence[nn.Module]
        Ensemble of critic networks. The input must be the state and action, and the output
        is the Q-value.

    value_net : nn.Module
        Value network. The input must be the state and the output is the value.

    actor_optimizer : Optimizer
        Optimizer for the actor network.

    critic_optimizer : Optimizer
        Optimizer for the critic networks.
    
    value_optimizer : Optimizer
        Optimizer for the value network.

    tau : float, optional
        Target network update rate. Default is 0.005.

    gamma : float, optional
        Discount factor. Default is 0.99.

    expectile : float, optional
        Expectile for the value loss. Default is 0.5.

    temperature : float, optional
        Temperature for the actor loss. Default is 0.1. 
    """

    def __init__(
            self,
            data: Buffer,
            actor: nn.Module,
            critic_ensemble: Sequence[nn.Module],
            value_net: nn.Module,
            actor_optimizer: Optimizer,
            critic_optimizer: Optimizer,
            value_optimizer: Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            expectile: float = 0.5,
            temperature: float = 0.1,
        ):
        super().__init__(data)

        self.actor                  = actor
        self.critic_ensemble        = critic_ensemble
        self.target_critic_ensemble = deepcopy(critic_ensemble)
        
        for critic in self.target_critic_ensemble:
            turn_off_grad(critic)


        self.value_net = value_net

        self.actor_optimizer  = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.value_optimizer  = value_optimizer

        self.tau = tau
        self.gamma = gamma
        self.expectile = expectile
        self.temperature = temperature
    
    def train(self, mode: bool = True) -> Self:
        self.training = mode

        self.actor.train(mode)

        for critic in self.critic_ensemble:
            critic.train(mode)

        self.value_net.train(mode)
    
        return self

    def train_step(self, batch: TensorDict) -> dict[str, Any]:
        obs           = batch['state']
        target_action = batch['action']
        rewards       = batch['reward']
        next_obs      = batch['next_state']
        dones         = batch['done']

        target_q_ensemble = torch.stack([
            critic(obs, target_action) for critic in self.target_critic_ensemble
        ], dim=-1)

        target_q_values = target_q_ensemble.min(dim=-1).values

        # Value update
        value = self.value_net(obs)

        advantages = target_q_values - value

        weight = torch.where(advantages > 0, self.expectile, 1 - self.expectile)

        value_loss = (weight * advantages**2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Critic update
        q_ensemble = torch.stack([
            critic(obs, target_action) for critic in self.critic_ensemble
        ], dim=-1)


        with torch.no_grad():
            target_values = self.value_net(next_obs)

            target_values = rewards + self.gamma * (~dones) * target_values
            target_values = target_values.unsqueeze(-1)


        critic_loss = ((q_ensemble - target_values)**2).mean(dim=0).sum()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Actor update
        advantages = advantages.detach()

        exp_advantages = torch.exp(advantages * self.temperature)
        exp_advantages = torch.clamp(exp_advantages, max=100.)

        _, log_prob = self.actor(obs, target_action)

        actor_loss = -(exp_advantages * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Target update
        for target, critic in zip(self.target_critic_ensemble, self.critic_ensemble):
            for target_param, param in zip(target.parameters(), critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return {
            "loss/actor" : actor_loss.item(),
            "loss/value" : value_loss.item(),
            "loss/critic" : critic_loss.item()
        }

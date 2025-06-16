from typing import Any, Literal
from collections.abc import Callable
from torch import Tensor

import random

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from gymnasium.vector import VectorEnv

from tensordict import TensorDict

from ..base import BaseAlgorithm
from ..buffer import Buffer
from ..protocols import Policy
from ..utils import get_device

Baselines = Literal['mean', 'greedy']

class Reinforce(BaseAlgorithm):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        env: VectorEnv[Any, Any, Any],
        actor: Policy[Tensor, Tensor],
        actor_optimizer: Optimizer,
        baseline: nn.Module | Baselines | None = None,
        /,
        mc_samples: int = 1,
        buffer_size: int = 10000,
        baseline_decay: float = 0.99,
        device: str = 'auto',
        grad_clip: float | None = None,
    ):
        self.obs_shape    = obs_shape
        self.action_shape = action_shape

        buffer = Buffer(
            buffer_size,
            buffer_shapes={
                'obs': obs_shape,
                'action': action_shape,
                'returns': (1,),
                'log_prob': (1,),
                'greedy_return': (1,)
            },
            device=get_device(device),
            allow_grad=False
        )

        super().__init__(buffer)

        self.env             = env
        self.actor           = actor
        self.actor_optimizer = actor_optimizer

        self.mc_samples      = mc_samples
        self.baseline        = baseline
        self.running_mean    = torch.tensor(0)
        self.baseline_decay  = baseline_decay
        self.grad_clip       = grad_clip

    def compute_baseline(self, batch: TensorDict) -> Tensor:
        if self.baseline is None:
            return torch.tensor(0)
    
        if isinstance(self.baseline, str):
            if self.baseline == 'mean':
                batch_mean = torch.mean(batch['returns'])

                self.running_mean = (
                    self.baseline_decay  * self.running_mean +
                    (1 - self.baseline_decay) * batch_mean
                )

                return self.running_mean.expand(batch['returns'].shape)

            if self.baseline == 'greedy':
                return batch['greedy_return']
        
            raise ValueError(f"Unknown baseline type: {self.baseline}")

        with torch.no_grad():
            return self.baseline(batch['obs'])

    def on_epoch_start(self) -> dict[str, Any]:
        self.buffer.clear()
        n_envs = self.env.num_envs

        all_greedy_returns = np.zeros((n_envs, self.mc_samples))

        for i in range(self.mc_samples):
            seed = random.randint(0, 2**31 - 1)

            obs, _ = self.env.reset(seed=seed)

            observations = torch.tensor(obs).reshape(n_envs, *self.obs_shape)

            actions, log_probs = self.actor.get_action(observations.to(self.device))
            actions = actions.reshape(n_envs, *self.action_shape)

            _, returns, *_ = self.env.step(actions.cpu().numpy())

            obs, _ = self.env.reset(seed=seed)
            greedy_action = self.actor.greedy(observations.to(self.device))

            _, greedy_returns, *_ = self.env.step(greedy_action.cpu().numpy())
            all_greedy_returns[:, i] = greedy_returns

            self.buffer.add(
                obs           = observations,
                action        = actions,
                returns       = torch.tensor(returns).reshape(n_envs, 1),
                log_prob      = log_probs,
                greedy_return = torch.tensor(greedy_returns).reshape(n_envs, 1)
            )

        return {
            'rewards' : all_greedy_returns
        }

    def update(self, batch: TensorDict) -> dict[str, Any]:
        returns   = batch['returns']
        log_probs = batch['log_prob']

        baseline = self.compute_baseline(batch)

        advantages = returns - baseline
        loss = torch.mean(-log_probs * advantages)

        self.actor_optimizer.zero_grad()
        loss.backward()

        self.actor_optimizer.step()

        return {
            "loss/actor": loss.item(),
        }

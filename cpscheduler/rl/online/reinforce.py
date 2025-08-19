from typing import Any, Literal, TypeAlias
from collections.abc import Callable
from torch.types import Tensor, Device

from math import sqrt
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

Baselines: TypeAlias = Literal["mean", "greedy", "none"] | Callable[[Tensor], Tensor]


class Reinforce(BaseAlgorithm):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        envs: VectorEnv[Any, Any, Any],
        actor: Policy[Tensor, Tensor],
        actor_optimizer: Optimizer,
        baseline: nn.Module | Baselines | None = None,
        /,
        norm_returns: bool = False,
        mc_samples: int = 1,
        n_steps: int = 1,
        device: Device = "auto",
    ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        buffer_size = n_steps * envs.num_envs * mc_samples

        buffer = Buffer(
            buffer_size,
            buffer_shapes={
                "obs": obs_shape,
                "action": action_shape,
                "returns": (1,),
                "greedy_return": (1,),
            },
            allow_grad=False,
        )

        super().__init__(buffer, actor, get_device(device))

        self.envs = envs
        self.actor_optimizer = actor_optimizer

        self.norm_returns = norm_returns
        self.mc_samples = mc_samples
        self.baseline = baseline

        self.running_mean = torch.tensor(0.0, device=self.device)
        self.sqrd_sum = torch.tensor(0.0, device=self.device)
        self.count = 0

    def compute_baseline(self, batch: TensorDict) -> Tensor:
        if self.baseline is None or self.baseline == "none":
            return torch.tensor(0)

        if isinstance(self.baseline, str):
            if self.baseline == "mean":
                return self.running_mean

            if self.baseline == "greedy":
                return batch["greedy_return"]

            raise ValueError(f"Unknown baseline type: {self.baseline}")

        with torch.no_grad():
            return self.baseline(batch["obs"])

    def on_epoch_start(self) -> dict[str, Any]:
        self.buffer.clear()
        n_envs = self.envs.num_envs

        all_greedy_returns = np.zeros((n_envs, self.mc_samples), dtype=np.float32)

        with torch.no_grad():
            for i in range(self.mc_samples):
                seed = random.randint(0, 2**31 - 1)

                obs, _ = self.envs.reset(seed=seed)

                observations = torch.tensor(obs).reshape(n_envs, *self.obs_shape)

                actions, _ = self.policy.get_action(observations.to(self.device))
                actions = actions.reshape(n_envs, *self.action_shape)

                _, returns, *_ = self.envs.step(actions.cpu().numpy())

                obs, _ = self.envs.reset(seed=seed)

                greedy_action = self.policy.greedy(observations.to(self.device))

                _, greedy_returns, *_ = self.envs.step(greedy_action.cpu().numpy())
                all_greedy_returns[:, i] = greedy_returns

                self.buffer.add(
                    obs=observations,
                    action=actions,
                    returns=torch.tensor(returns).reshape(n_envs, 1),
                    greedy_return=torch.tensor(greedy_returns).reshape(n_envs, 1),
                )

        batch_sum = np.sum(all_greedy_returns).item()
        n_samples = self.mc_samples * n_envs

        delta = batch_sum - self.running_mean

        new_count = self.count + n_samples
        self.running_mean = (self.count * self.running_mean + batch_sum) / new_count

        sqrd_diff = np.sum((all_greedy_returns - batch_sum / n_samples) ** 2).item()

        self.sqrd_sum += sqrd_diff + delta**2 * self.count * n_samples / new_count
        self.count = new_count

        return {
            "rewards": all_greedy_returns,
        }

    def update(self, batch: TensorDict) -> dict[str, Any]:
        returns = batch["returns"]

        log_probs = self.policy.log_prob(batch["obs"], batch["action"])

        baseline = self.compute_baseline(batch)

        advantages = returns - baseline

        if self.norm_returns:
            advantages /= self.sqrd_sum.sqrt()
            advantages *= sqrt(self.count - 1)

        loss = torch.mean(-log_probs * advantages)

        self.actor_optimizer.zero_grad()
        loss.backward()

        self.actor_optimizer.step()

        return {
            "loss/actor": loss.item(),
        }

    def on_session_end(self) -> None:
        super().on_session_end()

        self.envs.close()

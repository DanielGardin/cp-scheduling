from typing import Any, Optional, Callable

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .buffer import Buffer
from .base import BaseAlgorithm, Logs

from ..environment import Env, AsyncVectorEnv
# from ..utils import merge_and_pad

class PPO(BaseAlgorithm):
    def __init__(
            self,
            agent: nn.Module,
            value: nn.Module,
            optimizer: optim.Optimizer,
            env_fn: Callable[[], Env[Any, Any]],
            obs_shape: tuple[int, ...],
            action_shape: tuple[int, ...],
            *,
            n_envs: int = 4,
            num_steps: int = 128,
            anneal_lr: bool = True,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            norm_adv: bool = True,
            clip_loss: float = 0.2,
            clip_value: bool = True,
            value_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            target_kl: Optional[float] = 0.03,
        ):
        buffer_shapes = {
            'obs': obs_shape,
            'action': action_shape,
            'log_prob': (),
            'value': (),
            'returns': (),
            'advantages': (),
            'done': (),
        }

        buffer = Buffer(n_envs*num_steps, buffer_shapes)
        super().__init__(buffer)

        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.agent = agent
        self.value = value
        self.optimizer = optimizer
        self.env_fn = env_fn

        self.n_envs = n_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma

        self.gae_lambda = gae_lambda
        self.norm_adv = norm_adv
        self.clip_loss = clip_loss
        self.clip_value = clip_value
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
    

    def _compute_advantages(
            self,
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
            next_value: torch.Tensor
        ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)

        last_advantage = 0
        last_value = next_value

        for t in reversed(range(self.num_steps)):
            mask = 1 - dones[:, t]

            delta = rewards[:, t] + self.gamma * last_value * mask - values[:, t]

            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask

            advantages[:, t] = last_advantage
            last_value = values[:, t]

        return advantages


    def on_epoch_start(self) -> dict[str, Any] | Logs:
        envs = AsyncVectorEnv(
            [self.env_fn for _ in range(self.n_envs)]
        )

        obs, _ = envs.reset()

        capacity = self.num_steps * self.n_envs

        observations = torch.empty((self.n_envs, self.num_steps, *self.obs_shape), dtype=torch.float32)
        actions      = torch.empty((self.n_envs, self.num_steps, *self.action_shape), dtype=torch.int)
        rewards      = torch.empty((self.n_envs, self.num_steps), dtype=torch.float32)
        log_probs    = torch.empty((self.n_envs, self.num_steps), dtype=torch.float32)
        values       = torch.empty((self.n_envs, self.num_steps), dtype=torch.float32)
        dones        = torch.empty((self.n_envs, self.num_steps), dtype=torch.float32)

        cum_rewards = torch.zeros(self.n_envs, dtype=torch.float32)

        episode_rewards = []

        for step in range(self.num_steps):
            tensor_obs = torch.from_numpy(obs)

            with torch.no_grad():
                logits = self.agent(tensor_obs)
                value  = self.value(tensor_obs).flatten()

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())

            observations[:, step] = tensor_obs
            actions[:, step]      = action
            rewards[:, step]      = torch.tensor(reward)
            log_probs[:, step]    = dist.log_prob(action)
            values[:, step]       = value
            dones[:, step]        = torch.tensor(terminated)

            cum_rewards += torch.tensor(reward)

            if any(terminated):
                episode_rewards.extend(cum_rewards[terminated])
                cum_rewards[terminated] = 0

            obs = next_obs

        tensor_obs = torch.from_numpy(obs)
        with torch.no_grad():
            next_value = self.value(tensor_obs).flatten()

        advantages = self._compute_advantages(rewards, values, dones, next_value)

        self.buffer.add(
            obs        = observations.view(capacity, *self.obs_shape),
            action     = actions.view(capacity, *self.action_shape),
            log_prob   = log_probs.view(capacity,),
            value      = values.view(capacity,),
            returns    = (values + advantages).view(capacity,),
            advantages = advantages.view(capacity,),
            done       = dones.view(capacity,)
        )

        return Logs().extend({
            'episode_rewards': episode_rewards
        })
    

    def update(self, batch: TensorDict) -> dict[str, Any]:
        logits = self.agent(batch['obs'])

        dist = torch.distributions.Categorical(logits=logits)

        log_probs = dist.log_prob(batch['action'])
        entropy = dist.entropy().mean()

        log_ratio = log_probs - batch['log_prob']
        ratio = torch.exp(log_ratio)

        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        advantages = batch['advantages']

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_loss, 1 + self.clip_loss)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        value = self.value(batch['obs']).flatten()

        if self.clip_value:
            v_loss_unclipped = (value - batch['returns'])**2

            v_clipped = batch['value'] + torch.clamp(value - batch['value'], -self.clip_loss, self.clip_loss)

            v_loss_clipped = (v_clipped - batch['returns'])**2

            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        else:
            value_loss = 0.5 * F.mse_loss(value, batch['returns'])

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl': approx_kl.item()
        }


    def on_session_start(self, num_updates: int, steps_per_update: int, batch_size: int) -> None:
        if self.anneal_lr:
            self.lr_step = [group['lr']/num_updates for group in self.optimizer.param_groups]


    def on_epoch_end(self) -> dict[str, Any] | Logs:
        if self.anneal_lr:
            for param_group, lr_step in zip(self.optimizer.param_groups, self.lr_step):
                param_group['lr'] = max(param_group['lr'] - lr_step, 0)

        return {}
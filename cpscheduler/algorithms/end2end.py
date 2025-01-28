from typing import Any, Literal, Optional, Callable, Iterable, SupportsInt
from torch.types import Device, Tensor

import random

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from .base import BaseAlgorithm
from .buffer import Buffer

from ..environment.wrappers import End2EndStateWrapper
from ..environment.vector import AsyncVectorEnv, SyncVectorEnv, RayVectorEnv
from ..environment.protocols import VectorEnv

class End2End(BaseAlgorithm):
    def __init__(
        self,
        agent: nn.Module,
        optimizer: optim.Optimizer,
        env_fn: Callable[[], End2EndStateWrapper],
        n_envs: int = 128,
        n_jobs: int = 10,
        n_machines: int = 10,
        vector_env: Literal["async", "sync", "ray"] = "async",
        device: Device = "cuda",
        *,
        n_future_tasks: int = 3,
        clip_coef: float = 0.2,
        reward_norm: bool = True,
        target_kl: Optional[float] = None,
        anneal_lr: bool = True,
        time_limit: Optional[int] = 30,
    ):
        buffer_shapes = {
            "obs": (n_jobs, 2 + n_future_tasks, 6),
            "action": (),
            "log_prob": (),
            "improvement": (),
            "returns": (),
        }

        capacity = 2 * n_envs * n_jobs * n_machines
        buffer = Buffer(capacity, buffer_shapes, device)

        super().__init__(buffer)

        self.agent = agent
        self.optimizer = optimizer

        self.n_future_tasks = n_future_tasks
        self.clip_coef      = clip_coef
        self.reward_norm    = reward_norm
        self.anneal_lr      = anneal_lr
        self.target_kl      = float('inf') if target_kl is None else target_kl

        self.n_jobs     = n_jobs
        self.n_machines = n_machines
        self.env_fn = env_fn

        self.n_envs = n_envs
        self.vector_env = vector_env

        self.time_limit = time_limit


    def on_session_start(
        self, num_updates: int, steps_per_update: int, batch_size: int
    ) -> None:
        if self.anneal_lr:
            self.lr_delta = self.optimizer.param_groups[0]["lr"] / num_updates


    def on_epoch_start(self) -> dict[str, Any]:
        self.agent.eval()
        self.buffer.clear()

        envs: VectorEnv[Tensor, SupportsInt | Iterable[SupportsInt]]
        if self.vector_env == "async":
            envs = AsyncVectorEnv(
                [self.env_fn for _ in range(self.n_envs)], auto_reset=False
            )

        elif self.vector_env == "sync":
            envs = SyncVectorEnv(
                [self.env_fn for _ in range(self.n_envs)], auto_reset=False
            )

        else:
            envs = RayVectorEnv(
                [self.env_fn for _ in range(self.n_envs)], auto_reset=False
            )

        horizon = self.n_jobs * self.n_machines

        obs, info = envs.reset()

        observations = torch.empty(
            (self.n_envs, horizon, *self.buffer.buffer_shapes["obs"]),
            dtype=torch.float32,
        )
        actions = torch.empty(
            (self.n_envs, horizon, *self.buffer.buffer_shapes["action"]),
            dtype=torch.int64,
        )
        log_probs = torch.empty(
            (self.n_envs, horizon, *self.buffer.buffer_shapes["log_prob"]),
            dtype=torch.float32,
        )

        for i in range(horizon):
            tensor_obs = torch.stack(obs).to(self.device)
            observations[:, i] = tensor_obs

            with torch.no_grad():
                logits = self.agent(tensor_obs).cpu()

            categorical = torch.distributions.Categorical(logits=logits)

            action   = categorical.sample()
            log_prob = categorical.log_prob(action)

            actions[:, i]   = action
            log_probs[:, i] = log_prob

            obs, reward, terminated, truncated, info = envs.step(action)


        agent_return = torch.tensor(info["objective_value"], dtype=torch.float32)

        # Calculate partial solution with the cp solver to calculate improvement
        sampled_idx = random.randint(1, horizon-1)

        logged_actions = actions[:, :sampled_idx]

        envs.reset()

        obs, reward, terminated, truncated, info = envs.step(logged_actions)
        cp_actions, cp_start_time, cp_makespan, is_optimal = envs.call("get_cp_solution", timelimit=self.time_limit)

        cp_return = torch.tensor(cp_makespan, dtype=torch.float32)

        improvement = 1 - cp_return / agent_return
        return_ = cp_return.repeat(horizon, 1).T
        return_[:, sampled_idx:] = agent_return.unsqueeze(-1)

        self.buffer.add(
            obs         = observations,
            action      = actions,
            log_prob    = log_probs,
            improvement = -improvement.repeat(horizon, 1).T,
            returns     = return_,
        )

        new_horizon = horizon - sampled_idx
        for i in range(new_horizon):
            tensor_obs = torch.stack(obs).to(self.device)
            observations[:, i] = tensor_obs

            with torch.no_grad():
                logits = self.agent(tensor_obs).cpu()

            categorical = torch.distributions.Categorical(logits=logits)

            action = torch.tensor([
                cp_actions[job][i] for job in range(self.n_envs)
            ])

            log_prob = categorical.log_prob(action)

            assert torch.isfinite(log_prob).all()

            actions[:, i]   = action
            log_probs[:, i] = log_prob

            obs, reward, terminated, truncated, info = envs.step(action.cpu())

        self.buffer.add(
            obs         = observations[:, :new_horizon],
            action      = actions[:, :new_horizon],
            log_prob    = log_probs[:, :new_horizon],
            improvement = improvement.repeat(new_horizon, 1).T,
            returns     = cp_return.repeat(new_horizon, 1).T,
        )

        if self.reward_norm:
            self.buffer.normalize("returns", kind='standard')
            self.buffer.normalize("improvement")

        envs.close()

        return {
            "objective": agent_return.tolist(),
            "improvement": improvement.tolist(),
        }


    def update(self, batch: TensorDict) -> dict[str, Any]:
        logits = self.agent(batch["obs"])

        log_prob = torch.distributions.Categorical(logits=logits).log_prob(
            batch["action"]
        )

        log_ratio = log_prob - batch["log_prob"]
        ratio = torch.exp(log_ratio)

        improvement = batch["improvement"]

        improvement_loss1 = -improvement * ratio
        improvement_loss2 = -improvement * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        improvement_loss = torch.max(improvement_loss1, improvement_loss2).mean()

        improvement_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


        new_logits = self.agent(batch["obs"])
        new_log_prob = torch.distributions.Categorical(logits=new_logits).log_prob(
            batch["action"]
        )

        new_log_ratio = new_log_prob - batch["log_prob"]
        new_ratio     = torch.exp(new_log_ratio)

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl = torch.mean((ratio - 1) - new_log_ratio)

        if approx_kl > self.target_kl:
            raise ValueError(f"approx_kl is too high: {approx_kl}")

        returns = batch["returns"]

        pg_loss1 = returns * new_ratio
        pg_loss2 = returns * torch.clamp(new_ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        pg_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "improvement_loss": improvement_loss.item(),
            "return_loss" : pg_loss.item(),
            "approx_kl": approx_kl.item(),
        }

    def on_epoch_end(self) -> dict[str, Any]:
        epoch_lr = self.optimizer.param_groups[0]["lr"]

        if self.anneal_lr:
            self.optimizer.param_groups[0]["lr"] = max(epoch_lr - self.lr_delta, 1e-6)

        return {"learning rate": epoch_lr}


    # def validate(self) -> dict[str, Any] | Logs:
    #     makespans = list(map(
    #         int,
    #         [
    #             read_jsp_instance(root / f"instances/jobshop/ta{i:02d}.txt")[1]["Makespan UB"]
    #             for i in range(1, 81)
    #         ],
    #     ))

    #     val_makespans = []

    #     for group in range(8):
    #         initial = group * 10 + 1
    #         taillard_envs = AsyncVectorEnv(
    #             [make_eval_env(i) for i in range(initial, initial + 10)],
    #             auto_reset=False,
    #         )

    #         obs, info = taillard_envs.reset()
    #         running = [True] * 10
    #         while any(running):
    #             tensor_obs, tensor_mask = obs

    #             with torch.no_grad():
    #                 logits = self.agent(tensor_obs, tensor_mask)
    #                 action = torch.argmax(logits, dim=1)

    #             obs, reward, terminated, truncated, info = taillard_envs.step(
    #                 action.cpu().numpy()
    #             )

    #             running = [not done for done in terminated]

    #         val_makespans.extend(info["objective_value"])

    #     return Logs({
    #         "optimality gap": [
    #             (val - optimal) / optimal
    #             for val, optimal in zip(val_makespans, makespans)
    #         ]
    #     })
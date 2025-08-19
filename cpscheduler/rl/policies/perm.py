from collections.abc import Callable

from functools import lru_cache

from torch import Tensor

import torch
from torch import nn

from math import log

from cpscheduler.rl.protocols import Policy
from cpscheduler.heuristics._pdr import PriorityDispatchingRule


@lru_cache
def prob_to_lmbda(prob: float, size: int, n_iter: int) -> float:
    """
    Convert a probability to a lambda parameter for the Plackett-Luce model.
    """
    if prob == 1.0:
        return float("inf")

    x = 1 - prob
    for _ in range(n_iter):
        x = (prob * x**size * (size - 1) - (1 - prob)) / (
            size * prob * x ** (size - 1) - 1
        )

    return -log(x)


class PlackettLucePolicy(nn.Module, Policy[Tensor, Tensor]):
    def __init__(
        self,
        score_model: Callable[[Tensor], Tensor],
    ):
        super().__init__()
        self.score_model = score_model

    def get_score(self, x: Tensor) -> Tensor:
        logits = torch.squeeze(self.score_model(x), -1)

        return logits - torch.mean(logits, dim=-1, keepdim=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.sample(x)

    def greedy(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.get_score(x)

        permutation = torch.argsort(logits, dim=-1, descending=True)

        return permutation

    def sample(
        self,
        x: Tensor,
        temperature: float | Tensor = 1.0,
    ) -> tuple[Tensor, Tensor]:
        logits = self.get_score(x)
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(logits.shape)
        assert isinstance(gumbel_noise, Tensor)
        gumbel_noise = gumbel_noise.to(logits.device)

        if isinstance(temperature, Tensor):
            temperature = temperature.view(-1, 1).expand_as(logits)

        perturbed_logits = logits + temperature * gumbel_noise

        permutation = torch.argsort(
            perturbed_logits, descending=True, stable=True, dim=-1
        )

        log_prob = self.log_prob(x, permutation, temp=temperature)

        return permutation, log_prob

    def get_action(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.sample(x)

    def log_prob(self, x: Tensor, action: Tensor, temp: float | Tensor = 1.0) -> Tensor:
        scores = self.get_score(x) / temp
        scores = scores - scores.max(dim=-1, keepdim=True).values
        permuted_scores = scores.gather(-1, action)

        logcumsum = torch.logcumsumexp(permuted_scores.flip(dims=[-1]), dim=-1).flip(
            dims=[-1]
        )

        log_probs = torch.sum(permuted_scores - logcumsum, dim=-1)

        return log_probs

    def get_temperature(
        self, x: Tensor, target_prob: float, n_iter: int = 10
    ) -> Tensor:
        with torch.no_grad():
            logits = self.get_score(x)
            *batch, n_tasks = logits.shape

            if target_prob * n_tasks < 1:
                raise ValueError(
                    f"Target probability {target_prob} cannot be lower than uniform probability 1/{n_tasks}."
                )

            target_lmbda = prob_to_lmbda(target_prob, n_tasks, n_iter)

            X = torch.arange(n_tasks, device=logits.device, dtype=logits.dtype)
            X_centered = X - X.mean()

            ordered_logits, _ = torch.sort(logits, dim=-1, descending=True)
            centered_logits = ordered_logits - ordered_logits.mean(dim=-1, keepdim=True)

            cov_xy = torch.sum(X_centered * centered_logits, dim=-1)
            var_x = torch.sum(X_centered**2)

            lmbda = torch.reshape(-cov_xy / var_x, batch)

        return lmbda / target_lmbda

    def sample_pstar(
        self, x: Tensor, target_prob: float, n_iter: int = 10
    ) -> tuple[Tensor, Tensor]:
        """
        Sample a permutation with maximum probability bounded to a target probability.
        """
        temperature = self.get_temperature(x, target_prob, n_iter)

        return self.sample(x, temperature=temperature)


class WeightEstimationModel(nn.Module):
    def __init__(
        self,
        n_weights: int,
        feature_index: int = 0,
        preprocessor: nn.Module | None = None,
        pdr: PriorityDispatchingRule | None = None,
    ) -> None:
        super().__init__()
        self.feature_index = feature_index

        self.pdr = pdr
        self.preprocessor = preprocessor

        self.logit_weights = nn.Parameter(torch.zeros(n_weights))

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logit_weights, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_idxs = x[..., self.feature_index].long()

        x_weights = torch.gather(
            self.weights.expand(*weight_idxs.shape[:-1], -1), dim=-1, index=weight_idxs
        )

        if self.pdr is None:
            return x_weights

        obs = x.clone()
        obs[..., self.feature_index] = x_weights

        if self.preprocessor is not None:
            obs = self.preprocessor(obs)

        scores = self.pdr.get_priorities(obs)
        assert isinstance(scores, torch.Tensor)

        return scores.to(dtype=x_weights.dtype)

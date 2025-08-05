from collections.abc import Callable

from functools import lru_cache

from torch import Tensor

import torch
from torch import nn

from math import log

from cpscheduler.rl.protocols import Policy


@lru_cache
def prob_to_lmbda(prob: float, size: int, n_iter: int) -> float:
    """
    Convert a probability to a lambda parameter for the Plackett-Luce model.
    """
    if prob == 1.0:
        return float("inf")

    lmbda = 1 - prob
    for _ in range(n_iter):
        lmbda = (prob * lmbda**size * (size - 1) - (1 - prob)) / (
            size * prob * lmbda ** (size - 1) - 1
        )

    return lmbda


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

        permutation = torch.argsort(perturbed_logits, dim=-1, descending=True)

        log_prob = self.log_prob(x, permutation)

        return permutation, log_prob

    def get_action(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.sample(x)

    def log_prob(self, x: Tensor, action: Tensor) -> Tensor:
        scores = self.get_score(x)
        permuted_scores = scores.gather(-1, action)

        logcumsum = torch.logcumsumexp(permuted_scores.flip(dims=[-1]), dim=-1).flip(
            dims=[-1]
        )

        logits = torch.sum(permuted_scores - logcumsum, dim=-1)

        return logits

    def sample_pstar(
        self, x: Tensor, target_prob: float, n_iter: int = 10
    ) -> tuple[Tensor, Tensor]:
        """
        Sample a permutation with maximum probability bounded to a target probability.
        """
        logits = self.get_score(x)
        b, n = logits.shape

        target_lmbda = prob_to_lmbda(target_prob, n, n_iter)

        with torch.no_grad():
            ordered_logits, _ = torch.sort(logits, dim=-1, descending=True, stable=True)

            X = torch.stack([ordered_logits, torch.ones_like(ordered_logits)], dim=-1)
            y = -torch.arange(n, device=x.device).view(1, n, 1).expand(b, n, 1).float()

            XtX = X.transpose(-1, -2) @ X
            Xty = X.transpose(-1, -2) @ y

            coeffs: Tensor = torch.linalg.solve(XtX, Xty).squeeze(-1)

            lmbda = coeffs[:, 0]

        temperature = lmbda / target_lmbda

        return self.sample(x, temperature=temperature)

from collections.abc import Callable

from torch import Tensor

import torch
from torch import nn

from cpscheduler.rl.protocols import Policy


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
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        logits = self.get_score(x)
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)  # type: ignore

        perturbed_logits = logits / temperature + gumbel_noise

        permutation = torch.argsort(perturbed_logits, dim=-1, descending=True)

        log_prob = self.log_prob(x, permutation)

        return permutation, log_prob

    def get_action(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.sample(x)

    def log_prob(self, x: Tensor, action: Tensor) -> Tensor:
        scores = self.get_score(x)
        permuted_scores = scores.gather(1, action)

        logcumsum = torch.logcumsumexp(permuted_scores.flip(dims=[1]), dim=1).flip(
            dims=[1]
        )

        logits = torch.sum(permuted_scores - logcumsum, dim=1)

        return logits

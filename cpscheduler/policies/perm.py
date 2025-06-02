from typing import Any, Iterable, Callable, Optional, reveal_type

from torch import Tensor

import torch
from torch import nn

import math

class PlackettLucePolicy(nn.Module):
    def __init__(
        self,
        score_model: Callable[..., Tensor],
    ):
        super().__init__()
        self.score_model = score_model
    
    def forward(self, *x: Any) -> Tensor:
        logits = torch.squeeze(self.score_model(*x))

        return logits - torch.mean(logits, dim=-1, keepdim=True)

    def sample(
        self,
        *x: Any,
        temperature: float = 1.,
    ) -> tuple[Tensor, Tensor]:
        logits = self(*x)
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device) # type: ignore

        perturbed_logits = logits / temperature + gumbel_noise

        permutation = torch.argsort(perturbed_logits, dim=-1, descending=True)

        log_prob = self.log_prob(logits, permutation)

        return permutation, log_prob

    def greedy(self, *x: Tensor) -> Tensor:
        scores = self(*x)
        permutation = torch.argsort(scores, dim=-1, descending=True)

        return permutation


    def log_prob(
        self,
        scores: Tensor,
        permutations: Tensor
    ) -> Tensor:
        """
        Computes the log probability of a given permutation under the Plackett-Luce model.

        Parameters
        ----------
        scores : Tensor
            The scores for each item in the permutation.
        permutations : Tensor
            The permutations to compute the log probability for.

        Returns
        -------
        Tensor
            The log probability of the given permutation.
        """
        permuted_scores = scores.gather(1, permutations)

        logcumsum = torch.logcumsumexp(permuted_scores.flip(dims=[1]), dim=1).flip(dims=[1])

        logits = torch.sum(permuted_scores - logcumsum, dim=1)

        return logits
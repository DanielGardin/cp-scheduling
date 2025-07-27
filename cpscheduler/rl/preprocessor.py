from typing import Any
from collections.abc import Sequence
from typing_extensions import Self

from torch import Tensor

import torch
from torch import nn

from .utils import mean_features


class StandardNormLayer(nn.Module):
    mean: Tensor
    std: Tensor

    def __init__(self, size: Sequence[int], eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))

        self.eps = eps

    def fit(self, x: Tensor) -> Self:
        self.mean = mean_features(x)
        self.std = x.std(dim=tuple(range(0, x.ndim - 1)), unbiased=False)

        return self

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + self.eps)


class RunningStandardNormLayer(nn.Module):
    """
    A running standard normalization layer that updates its mean and std using
    Welford's online algorithm.
    """

    mean: Tensor
    count: Tensor
    mean_squared: Tensor

    def __init__(self, size: Sequence[int]):
        super().__init__()

        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("mean_squared", torch.zeros(size))

    @property
    def std(self) -> Tensor:
        return self.mean_squared.sqrt() / (self.count - 1).sqrt()

    def fit(self, x: Tensor) -> Self:
        batch_size = x.size(0)
        batch_mean = mean_features(x)

        total_count = self.count + batch_size
        delta = batch_mean - self.mean

        new_mean = self.count * self.mean + batch_mean * batch_size

        self.mean = new_mean / total_count

        squared_diff = (x - batch_mean).pow(2).sum(dim=0)

        self.mean_squared += (
            squared_diff + delta.pow(2) * self.count * batch_size / total_count
        )
        self.count = total_count

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self.fit(x)

        return (x - self.mean) * (self.count - 1).sqrt() / self.mean_squared.sqrt()


class TabularPreprocessor(nn.Module):
    """
    A preprocessor for tabular data that applies categorical encoding and
    standard normalization.
    """

    def __init__(
        self,
        categorical_indices: Sequence[int],
        numerical_indices: Sequence[int],
        categorical_embedding_dim: int,
    ):
        super().__init__()
        self.categorical_encodings = nn.ModuleList()
        self.numerical_norm = StandardNormLayer(size=(len(numerical_indices),))

        self.categorical_embedding_dim = categorical_embedding_dim

        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices

        self.output_dim = (
            len(numerical_indices)
            + len(categorical_indices) * categorical_embedding_dim
        )

    def fit(self, x: Tensor) -> Self:
        if not self.categorical_encodings:
            for index in self.categorical_indices:
                encoding = nn.Embedding(
                    num_embeddings=int(x[..., index].max().item() + 1),
                    embedding_dim=self.categorical_embedding_dim,
                )

                self.categorical_encodings.append(encoding)

        self.numerical_norm.fit(x[..., self.numerical_indices])

        return self

    def forward(self, x: Tensor) -> Tensor:
        num_x = self.numerical_norm(x[..., self.numerical_indices])

        cat_x = [
            encoding(x[..., index].long())
            for index, encoding in zip(
                self.categorical_indices, self.categorical_encodings
            )
        ]

        return torch.cat([num_x] + cat_x, dim=-1)

from typing import Any, Iterable, Callable, Optional

from torch import Tensor

import torch
from torch import nn

import math

def layer_init(layer: nn.Module, gain: float = 1, bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, gain) # type: ignore

    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const) # type: ignore

    return layer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.pe = pe

    def forward(self, positions: Tensor) -> Tensor:
        return self.pe[positions]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
    ):
        layer_dims = [input_dim] + list(hidden_dims) + [output_dim]

        super().__init__()
        self.layers = nn.ModuleList([
            layer_init(nn.Linear(layer_dims[i], layer_dims[i + 1]), gain=math.sqrt(2))
            for i in range(len(layer_dims) - 1)
        ])

        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

        self.batch_norm = nn.ModuleList([
            nn.BatchNorm1d(layer_dims[i + 1])
            for i in range(len(layer_dims) - 2)
        ]) if batch_norm else None

        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(layer_dims[i + 1])
            for i in range(len(layer_dims) - 2)
        ]) if layer_norm else None

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

            if self.batch_norm is not None:
                x = self.batch_norm[i](x)
            
            elif self.layer_norm is not None:
                x = self.layer_norm[i](x)

            x = self.activation(x)
            x = self.dropout(x)

        x = self.layers[-1](x)

        return x
    

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 2
        ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, num_items, feature_dim)
        x = self.transformer(x)
        return self.output(x).squeeze(-1)
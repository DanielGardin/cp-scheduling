from typing import Any
from torch import Tensor


import torch
from torch import nn

import math

def layer_init(layer: nn.Module, gain: int = 1, bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, gain)

    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    
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

from typing import Any, Optional, MutableSequence, TypeAlias, Callable
from torch.types import Device

from pathlib import Path

import numpy as np

from datetime import datetime

import torch
from torch.nn import Module

def soft_update(
    target: Module,
    source: Module,
    tau: float = 1.0,
) -> None:
    """
    Soft update of the target network parameters.

    Parameters
    ----------
    target : Module
        The target network.
    source : Module
        The source network.
    tau : float, optional
        The interpolation factor, by default 1.0
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
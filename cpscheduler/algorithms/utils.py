from typing import Any, Sequence, Callable

from pathlib import Path

import torch.optim as optim
from torch.nn import Module
import torch

def set_seed(seed: int) -> None:
    import random, torch
    import numpy as np

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def turn_off_grad(model: Module) -> None:
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
from torch.types import Device

from torch.nn import Module
import torch


def get_device(device: Device = "auto") -> Device:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device


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

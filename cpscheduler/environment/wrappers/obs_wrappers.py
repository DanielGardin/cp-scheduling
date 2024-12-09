from typing import Any
from torch.types import Device

from torch import as_tensor

from .wrapper import wrap_obs
from ..env import Env

def pytorch_wrapper(env: Env, device: Device) -> Env:
    return wrap_obs(
        env,
        lambda obs: as_tensor,
        device=device
    )
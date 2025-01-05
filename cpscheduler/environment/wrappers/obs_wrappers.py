from typing import Any, TypeVar, SupportsFloat
from torch.types import Device

from torch import Tensor, as_tensor

from .wrapper import WrappedEnv
from .. import Env


_Act = TypeVar('_Act')
class PytorchWrapper(WrappedEnv[Tensor, _Act]):
    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        obs, info = super().reset()

        return as_tensor(obs), info

    def step(self, action: _Act, *args: Any, **kwargs: Any) -> tuple[Tensor, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action, *args, **kwargs)

        return as_tensor(obs), reward, terminated, truncated, info
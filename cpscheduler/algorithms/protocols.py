from typing import Any, Optional, MutableSequence, TypeAlias, Callable, Protocol, TypeVar
from torch.types import Device, Tensor

from pathlib import Path

import numpy as np

from datetime import datetime

import torch
from torch.nn import Module
from tensordict import TensorDict
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter # type: ignore

from abc import ABC, abstractmethod

import logging
import tqdm

from .buffer import Buffer
from .utils import set_seed


ObsT_ = TypeVar("ObsT_", contravariant=True)
ActT_ = TypeVar("ActT_")

class Policy(Protocol[ObsT_, ActT_]):
    def get_action(self, x: ObsT_) -> tuple[ActT_, Tensor]:
        ...

    def log_prob(self, x: ObsT_, action: ActT_) -> Tensor:
        ...

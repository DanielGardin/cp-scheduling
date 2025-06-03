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


ObsT_C = TypeVar("ObsT_C", contravariant=True)
ActT_ = TypeVar("ActT_")

class Policy(Protocol[ObsT_C, ActT_]):
    def get_action(self, x: ObsT_C) -> tuple[ActT_, Tensor]:
        ...

    def log_prob(self, x: ObsT_C, action: ActT_) -> Tensor:
        ...

    def greedy(self, x: ObsT_C) -> ActT_:
        ...


ActT_C = TypeVar("ActT_C", contravariant=True)
class Critic(Protocol[ObsT_C, ActT_C]):
    def get_value(self, x: ObsT_C, action: ActT_C) -> Tensor:
        ...

    def get_values(self, x: ObsT_C) -> Tensor:
        ...

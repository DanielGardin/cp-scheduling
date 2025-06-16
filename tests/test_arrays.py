import pytest
from typing import Any

import numpy as np
import torch

from gymnasium import Env

from common import env_setup

from cpscheduler.environment.wrappers import PermutationActionWrapper

def test_numpy_scalar() -> None:
    env = env_setup("ta01")

    env.reset()
    obs, reward, terminated, truncated, info = env.step(("execute", np.array(0)))

def test_numpy_array() -> None:
    env = PermutationActionWrapper(
        env_setup("ta01")
    )

    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.array([0, 1, 2, 3]))

def test_torch_scalar() -> None:
    env = env_setup("ta01")

    env.reset()
    obs, reward, terminated, truncated, info = env.step(("execute", torch.tensor(0)))

def test_torch_array() -> None:
    env = PermutationActionWrapper(
        env_setup("ta01")
    )

    env.reset()

    obs, reward, terminated, truncated, info = env.step(torch.tensor([0, 1, 2, 3]))

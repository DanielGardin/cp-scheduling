"""
    env.py

This module defines the `SchedulingEnvGym` class, which is a Gymnasium environment wrapper for
the CPScheduler scheduling environment.
"""

from typing import Any
from collections.abc import Iterable

from gymnasium import Env
from gymnasium.spaces import Tuple, Text, Box, OneOf, Sequence

import numpy as np

from cpscheduler.environment._common import MAX_INT, InstanceConfig, ObsType
from cpscheduler.environment.instructions import ActionType
from cpscheduler.environment import SchedulingEnv, ScheduleSetup, Constraint, Objective
from cpscheduler.environment._render import Renderer

from .gym_utils import infer_collection_space

# Define the action space for the environment
InstructionSpace = Text(max_length=10)
IntSpace = Box(low=0, high=int(MAX_INT), shape=(), dtype=np.int64)

SingleActionSpace = OneOf(
    [
        Tuple([InstructionSpace]),
        Tuple([InstructionSpace, IntSpace]),
        Tuple([InstructionSpace, IntSpace, IntSpace]),
        Tuple([InstructionSpace, IntSpace, IntSpace, IntSpace]),
    ]
)

ActionSpace = Sequence(SingleActionSpace, stack=True)


class SchedulingEnvGym(Env[ObsType, ActionType]):
    metadata: dict[str, Any] = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        machine_setup: ScheduleSetup,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        instance_config: InstanceConfig | None = None,
        *,
        render_mode: Renderer | str | None = None,
        n_parts: int = 1,
    ):
        self.action_space = ActionSpace

        self._env = SchedulingEnv(
            machine_setup=machine_setup,
            constraints=constraints,
            objective=objective,
            instance_config=instance_config,
            render_mode=render_mode,
            n_parts=n_parts,
        )

        self.observation_space = infer_collection_space(self._env._get_state())

    @classmethod
    def from_env(cls, env: SchedulingEnv) -> "SchedulingEnvGym":
        """
        Create a `SchedulingEnvGym` instance from an existing `SchedulingEnv`.
        """
        self = cls.__new__(cls)

        self.action_space = ActionSpace
        self._env = env

        return self

    @property
    def core(self) -> SchedulingEnv:
        return self._env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | InstanceConfig | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        previously_loaded = self._env.loaded

        obs, info = self._env.reset(
            options=options,
        )

        if options is not None or not previously_loaded:
            self.observation_space = infer_collection_space(obs)

        return obs, {key: value for key, value in info.items()}

    def step(
        self, action: ActionType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = self._env.step(action)

        return obs, reward, done, truncated, {key: value for key, value in info.items()}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def __repr__(self) -> str:
        return self._env.__repr__()

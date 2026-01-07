"""
    env.py

This module defines the `SchedulingEnvGym` class, which is a Gymnasium environment wrapper for
the CPScheduler scheduling environment.
"""

from typing import Any
from collections.abc import Iterable, Mapping

from gymnasium import Env, Space

from cpscheduler.utils._protocols import Metric

from cpscheduler.environment import SchedulingEnv, ScheduleSetup, Constraint, Objective
from cpscheduler.environment._common import InstanceTypes, ObsType, Options
from cpscheduler.environment.instructions import ActionType
from cpscheduler.environment._render import Renderer

from .gym_utils import infer_collection_space
from .common import ActionSpace


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
        instance: InstanceTypes | None = None,
        metrics: Mapping[str, Metric[Any]] | None = None,
        *,
        render_mode: Renderer | str | None = None,
        allow_preemption: bool = False,
    ):
        self.action_space = ActionSpace

        self._env = SchedulingEnv(
            machine_setup=machine_setup,
            constraints=constraints,
            objective=objective,
            instance=instance,
            metrics=metrics,
            render_mode=render_mode,
            allow_preemption=allow_preemption,
        )

        self.observation_space = self.get_observation_space()

    def get_observation_space(self) -> Space[ObsType]:
        return infer_collection_space(self._env.get_state())

    @classmethod
    def from_env(cls, env: SchedulingEnv) -> "SchedulingEnvGym":
        "Create a `SchedulingEnvGym` instance from an existing `SchedulingEnv`."
        self = cls.__new__(cls)

        self.action_space = ActionSpace
        self._env = env

        return self

    @property
    def core(self) -> SchedulingEnv:
        "Return the underlying `SchedulingEnv` instance."
        return self._env

    def reset(
        self, *, seed: int | None = None, options: Options = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        previously_loaded = self._env.state.loaded

        obs, info = self._env.reset(options=options)

        if options is not None or not previously_loaded:
            self.observation_space = self.get_observation_space()

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

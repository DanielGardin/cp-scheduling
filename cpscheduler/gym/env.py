"""
    env.py

This module defines the `SchedulingEnvGym` class, which is a Gymnasium environment wrapper for
the CPScheduler scheduling environment.
"""

from collections.abc import Iterable, Mapping
from typing import Any, cast, overload

from gymnasium import Env, Space
from typing_extensions import TypeVar

from cpscheduler.environment import (
    Constraint,
    Objective,
    ScheduleSetup,
    SchedulingEnv,
)
from cpscheduler.environment.des import ActionType
from cpscheduler.environment.observation import Observation
from cpscheduler.environment.observation.default import DefaultObsType
from cpscheduler.environment.render import Renderer
from cpscheduler.environment.utils.protocols import (
    InstanceTypes,
    Metric,
    Options,
)
from cpscheduler.gym.common import ActionSpace
from cpscheduler.gym.obs_spaces import observation_spec_to_gym_space

ObsType = TypeVar("ObsType", default=DefaultObsType)


class SchedulingEnvGym(Env[ObsType, ActionType]):
    _core: SchedulingEnv[Observation[ObsType]]

    @overload
    def __init__(
        self: "SchedulingEnvGym[DefaultObsType]",
        machine_setup: ScheduleSetup,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        observation: None = None,
        instance: InstanceTypes | None = None,
        metrics: Mapping[str, Metric[Any]] | None = None,
        *,
        render_mode: Renderer | str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        machine_setup: ScheduleSetup,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        *,
        observation: Observation[ObsType],
        instance: InstanceTypes | None = None,
        metrics: Mapping[str, Metric[Any]] | None = None,
        render_mode: Renderer | str | None = None,
    ) -> None: ...

    def __init__(
        self,
        machine_setup: ScheduleSetup,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        observation: Observation[ObsType] | None = None,
        instance: InstanceTypes | None = None,
        metrics: Mapping[str, Metric[Any]] | None = None,
        *,
        render_mode: Renderer | str | None = None,
    ):
        self.action_space = ActionSpace

        if observation is None:
            _env = SchedulingEnv(
                machine_setup=machine_setup,
                constraints=constraints,
                objective=objective,
                instance=instance,
                metrics=metrics,
                render_mode=render_mode,
            )

            self._core = cast(SchedulingEnv[Observation[ObsType]], _env)

        else:
            self._core = SchedulingEnv(
                machine_setup=machine_setup,
                constraints=constraints,
                objective=objective,
                observation=observation,
                instance=instance,
                metrics=metrics,
                render_mode=render_mode,
            )

        self.observation_space = self._get_observation_space()
        self.metadata = {
            "render_modes": ["human"],
            "render_fps": 50,
        }

    def _get_observation_space(self) -> Space[Any]:
        return observation_spec_to_gym_space(self._core.observation)

    @classmethod
    def from_env(
        cls, env: SchedulingEnv[Observation[ObsType]]
    ) -> "SchedulingEnvGym[ObsType]":
        "Create a `SchedulingEnvGym` instance from an existing `SchedulingEnv`."
        self = cls.__new__(cls)
        super().__init__(self)

        self.action_space = ActionSpace
        self._core = env

        self.observation_space = self._get_observation_space()
        self.metadata = {
            "render_modes": ["human"],
            "render_fps": 50,
        }

        return self

    @property
    def core(self) -> SchedulingEnv[Observation[ObsType]]:
        "Return the underlying `SchedulingEnv` instance."
        return self._core

    def reset(
        self, *, seed: int | None = None, options: Options | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        obs, info = self._core.reset(options=options)

        # FUTURE: This is expensive, highly unnecessary, but fine for now.
        self.observation_space = self._get_observation_space()

        return obs.serialize(), {key: value for key, value in info.items()}

    def step(
        self, action: ActionType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = self._core.step(action)

        return (
            obs.serialize(),
            reward,
            done,
            truncated,
            {key: value for key, value in info.items()},
        )

    def __repr__(self) -> str:
        return self._core.__repr__()

    # Expose SchedulingEnv public methods

    def load_instance(self, instance: InstanceTypes) -> None:
        """Set problem instance data."""
        self._core.load_instance(instance)

    def add_constraint(self, constraint: Constraint) -> None:
        self._core.add_constraint(constraint)

    def set_objective(self, objective: Objective) -> None:
        self._core.set_objective(objective)

    def add_metric(self, name: str, metric: Metric[Any]) -> None:
        self._core.add_metric(name, metric)

    def get_entry(self) -> str:
        return self._core.get_entry()

"""Gymnasium wrapper for the SchedulingEnv environment.

This module defines the `SchedulingEnvGym` class, which is a Gymnasium environment
wrapper for the main scheduling environment.
The core environment implements the main logic, this wrapper allows using
the gymnasium infrastructure and semantics without friction.
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
from cpscheduler.environment.utils import InstanceGenerator
from cpscheduler.environment.utils.protocols import (
    Instance_T,
    InstanceTypes,
    Metric,
    Options,
)
from cpscheduler.gym.common import ActionSpace
from cpscheduler.gym.obs_spaces import observation_spec_to_gym_space

ObsType = TypeVar("ObsType", default=DefaultObsType)


class SchedulingEnvGym(Env[ObsType, ActionType]):
    """Reinforcement Learning environment for scheduling problems.

    SchedulingEnvGym wraps the composition of machine setup (alpha),
    constraints (beta), and objectives (gamma) into a RL-friendy interface.
    """

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
        """Initialize the scheduling environment.

        Defines the parameterized environment, composing machine_setup,
        constraints and objective.

        If `instance` is an `InstanceGenerator`, it is stored for deferred
        sampling after each `reset()` call.
        If it is concrete instance data, `load_instance()` is called immediately.

        Parameters
        ----------
        machine_setup : ScheduleSetup, optional
            Alpha component in Graham notation. Defines machine topology and
            availability. Defaults to a bare `ScheduleSetup` (no machines).

        constraints : Iterable[Constraint], optional
            Beta components. Applied during constraint propagation.
            Defaults to an empty tuple.

        objective : Objective, optional
            Gamma component. Defines the reward signal. Defaults to a no-op
            `Objective`.

        observation : ObsT_co, optional
            Observation class used to build RL observations. Defaults to
            `DefaultObservation`.

        instance : InstanceTypes or InstanceGenerator, optional
            Either concrete instance data or a generator stored for lazy sampling.

        metrics : Mapping[str, Metric], optional
            Performance metrics to be added to the info dictionary.

        render_mode : Renderer or str, optional
            Renderer instance or mode string. Defaults to a no-op renderer.

        """
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

            self._core = cast("SchedulingEnv[Observation[ObsType]]", _env)

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
        """Create a `SchedulingEnvGym` instance from an existing `SchedulingEnv`."""
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
        """Return the underlying `SchedulingEnv` instance."""
        return self._core

    def reset(
        self, *, seed: int | None = None, options: Options | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment to its initial state for a new episode.

        Clears the schedule, resets all state variables, and performs initial
        constraint propagation. Optionally loads a new instance via generator.

        Parameters
        ----------
        seed: int, optional
            Seed for the instance generator and any stochastic components.

        options : dict, optional
            Configuration options including:
            - 'instance': Load a specific instance.
            - 'instance_generator': Replace the instance generator.

        Returns
        -------
        observation : ObsT_co
            Initial observation.

        info : dict[str, Any]
            Environment info (time, objective value, event count, etc.).

        Raises
        ------
        ValueError
            If no instance is loaded.

        RuntimeError
            If propagation detects initial inconsistency.

        """
        super().reset(seed=seed)

        if seed is not None:
            options = options or {}
            options["seed"] = seed

        obs, info = self._core.reset(options=options)

        # FUTURE: This is expensive, highly unnecessary, but fine for now.
        self.observation_space = self._get_observation_space()

        return obs.serialize(), info

    def step(
        self, action: ActionType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Execute one simulation step.

        Schedules the action, advances time, processes instructions, and updates
        the observation and reward.
        Note that a step in the environment can correspond to many unit time
        steps in the environment's clock.

        A step here is defined as the state advancement where one of the
        following conditions is met first:
        - The instruction queue is executed entirely.
        - A terminal state is reached.
        - A infeasibility is detected (a bad action caused a contradiction)

        Parameters
        ----------
        action : ActionType, optional
            Task assignment(s) or None.

        Returns
        -------
        observation : ObsT_co
            Current observation.

        reward : float
            Signed reward (Delta objective value).

        terminated : bool
            Whether all tasks are completed.

        truncated : bool
            Whether the episode reached an infeasible state.

        info : dict[str, Any]
            Environment info (time, objective, event count, etc.).

        Raises
        ------
        RuntimeError
            If environment is not in RUNNING state.
            Ensure you called `reset()` before `step()`.

        """
        obs, reward, done, truncated, info = self._core.step(action)

        return (
            obs.serialize(),
            reward,
            done,
            truncated,
            info,
        )

    def __repr__(self) -> str:
        """Return a string representation of the environment's configuration and state."""
        return self._core.__repr__()

    # Expose SchedulingEnv public methods

    def set_generator(self, instance: InstanceGenerator) -> None:
        """Set the instance generator.

        The instance generator will sample a new instance for every reset call
        """
        self._core.set_generator(instance)

    def load_instance(self, *instances: Instance_T) -> None:
        """Load a scheduling instance and initialize the environment.

        Prepares the environment for simulation by loading instance data,
        validating constraints, and propagating domain bounds.

        Parameters
        ----------
        *instances : InstanceTypes
            One or more instance data objects to load.
            If multiple instances are provided, they are merged.
            Allows for heterogeneous instance data sources.

        Raises
        ------
        ValueError
            If setup constraints produce invalid features.

        RuntimeError
            If constraint propagation detects initial infeasibility.

        """
        self._core.load_instance(*instances)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the environment.

        Parameters
        ----------
        constraint : Constraint
            The constraint to add.

        Notes
        -----
        Resets any loaded instance to allow configuration changes.

        """
        self._core.add_constraint(constraint)

    def set_objective(self, objective: Objective) -> None:
        """Replace the objective function.

        Parameters
        ----------
        objective : Objective
            The new objective function.

        Notes
        -----
        Resets any loaded instance to allow configuration changes.

        """
        self._core.set_objective(objective)

    def add_metric(self, name: str, metric: Metric[Any]) -> None:
        """Add a metric to the environment."""
        self._core.add_metric(name, metric)

    def get_entry(self) -> str:
        """Get a string representation of the environment's configuration."""
        return self._core.get_entry()

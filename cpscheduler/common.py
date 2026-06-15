"""Common helper functions for all modules."""

from importlib.machinery import EXTENSION_SUFFIXES
from typing import Any

from cpscheduler.environment.env import SchedulingEnv


def is_compiled() -> bool:
    """Return whether the environment module is compiled."""
    import cpscheduler.environment.env as env

    return any(env.__file__.endswith(suffix) for suffix in EXTENSION_SUFFIXES)


AnySchedulingEnv = Any | SchedulingEnv[Any]


def unwrap_env(
    env: AnySchedulingEnv, max_depth: int = 10
) -> SchedulingEnv[Any]:
    """Unwraps the environment to get the underlying SchedulingEnv instance.

    Parameters
    ----------
    env: Env | SchedulingEnv
        The environment to unwrap.

    max_depth: int, default=10
        Maximum number of unwraps for Gymansium environments until raisingg an error.

    Returns
    -------
    SchedulingEnv
        The unwrapped SchedulingEnv instance.

    Raises
    ------
    ValueError
        If the resulting environment is not a valid Gymnasium environment, or
        when no SchedulingEnv could be find up to max_depth.

    """
    depth = 0
    while not isinstance(env, SchedulingEnv):
        if hasattr(env, "core") and isinstance(env.core, SchedulingEnv):
            return env.core

        if hasattr(env, "unwrapped"):
            env = env.unwrapped
            depth += 1

        else:
            raise ValueError(
                f"Expected env to have 'unwrapped' or 'core' attribute, "
                f"got {type(env)} instead."
            )

        if depth >= max_depth:
            raise ValueError(
                "Maximum depth reached, could not find a SchedulingEnv from "
                f"the environment {type(env)}."
            )

    return env

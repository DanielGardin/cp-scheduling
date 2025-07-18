from typing import Any

from cpscheduler.environment.env import SchedulingEnv

MAX_ENV_DEPTH = 10  # Maximum depth for the environment wrapping


def unwrap_env(env: Any | SchedulingEnv) -> SchedulingEnv:
    """
    Unwraps the environment to get the underlying SchedulingEnv instance.

    Parameters
    ----------
    env: Env | SchedulingEnv
        The environment to unwrap.

    Returns
    -------
    SchedulingEnv
        The unwrapped SchedulingEnv instance.
    """
    depth = 0
    while not isinstance(env, SchedulingEnv) and depth < MAX_ENV_DEPTH:
        if not hasattr(env, "unwrapped"):
            raise TypeError(
                f"Expected env to be of type SchedulingEnv or a Wrapped env, got {type(env)} instead."
            )

        if hasattr(env, "core"):
            env = env.core
            break

        env = env.unwrapped
        depth += 1

    if not isinstance(env, SchedulingEnv):
        raise TypeError(
            f"Expected env to be of type SchedulingEnv, got {type(env)} instead."
        )

    return env

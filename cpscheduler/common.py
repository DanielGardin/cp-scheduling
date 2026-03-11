from typing import Any

from cpscheduler.environment.env import SchedulingEnv


def is_compiled() -> bool:
    import cpscheduler.environment.env as _

    return _.__file__.endswith(".so") or _.__file__.endswith(".pyd")


def unwrap_env(env: Any | SchedulingEnv, max_depth: int = 10) -> SchedulingEnv:
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
    while not isinstance(env, SchedulingEnv) and depth < max_depth:
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

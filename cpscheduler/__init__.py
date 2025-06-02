from gymnasium import register

from .environment import *

register(
    id="Scheduling-v0",
    entry_point="cpscheduler.environment:SchedulingEnv",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False
)
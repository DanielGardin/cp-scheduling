from gymnasium import register

from .environment import *
from . import aliases

register(
    id="Scheduling-v0",
    entry_point="cpscheduler.environment:SchedulingEnv",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False
)

register(
    id="Jobshop-v0",
    entry_point="cpscheduler.aliases:make_jobshop",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False
)

"""Gymnasium environments for scheduling problems."""

from gymnasium import register

__all__ = ["SchedulingEnvGym"]

from .env import SchedulingEnvGym

register(
    id="Scheduling-v0",
    entry_point="cpscheduler.gym:SchedulingEnvGym",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id="Jobshop-v0",
    entry_point="cpscheduler.gym.aliases:make_jobshop",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

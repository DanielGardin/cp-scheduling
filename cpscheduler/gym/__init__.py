"""Gymnasium environments for scheduling problems."""

from gymnasium import register

__all__ = ["SchedulingEnvGym"]

from .env import SchedulingEnvGym

# Jm||Cmax
register(
    id="Jobshop-v0",
    entry_point="cpscheduler.gym.aliases:make_jobshop",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Jm|rj|Cmax
register(
    id="JobshopDynamic-v0",
    entry_point="cpscheduler.gym.aliases:make_jobshop_dynamic",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Jm|rj|ΣTj
register(
    id="JobshopTardiness-v0",
    entry_point="cpscheduler.gym.aliases:make_jobshop_tardiness",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Jm|brkdwn|Cmax
register(
    id="JobshopBreakdown-v0",
    entry_point="cpscheduler.gym.aliases:make_jobshop_breakdown",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# FJm||Cmax
register(
    id="FlexibleJobshop-v0",
    entry_point="cpscheduler.gym.aliases:make_flexible_jobshop",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# FJm|rj|Cmax
register(
    id="FlexibleJobshopDynamic-v0",
    entry_point="cpscheduler.gym.aliases:make_flexible_jobshop_dynamic",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Fm||Cmax
register(
    id="Flowshop-v0",
    entry_point="cpscheduler.gym.aliases:make_flowshop",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Om||Cmax
register(
    id="Openshop-v0",
    entry_point="cpscheduler.gym.aliases:make_openshop",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Pm||Cmax
register(
    id="ParallelMakespan-v0",
    entry_point="cpscheduler.gym.aliases:make_parallel_makespan",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Pm||ΣCj
register(
    id="ParallelTotalCompletion-v0",
    entry_point="cpscheduler.gym.aliases:make_parallel_total_completion",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Pm||Σw_jCj
register(
    id="ParallelWeightedCompletion-v0",
    entry_point="cpscheduler.gym.aliases:make_parallel_weighted_completion",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Pm|rj|Cmax
register(
    id="ParallelDynamicMakespan-v0",
    entry_point="cpscheduler.gym.aliases:make_parallel_dynamic_makespan",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Pm|rj,M_j|Cmax
register(
    id="ParallelEligibilityMakespan-v0",
    entry_point="cpscheduler.gym.aliases:make_parallel_eligibility_makespan",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Rm||Cmax
register(
    id="UnrelatedMakespan-v0",
    entry_point="cpscheduler.gym.aliases:make_unrelated_makespan",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# Rm||Σw_jCj
register(
    id="UnrelatedWeightedCompletion-v0",
    entry_point="cpscheduler.gym.aliases:make_unrelated_weighted_completion",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# 1||ΣTj
register(
    id="SingleTotalTardiness-v0",
    entry_point="cpscheduler.gym.aliases:make_single_total_tardiness",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# 1||Σw_jTj
register(
    id="SingleWeightedTardiness-v0",
    entry_point="cpscheduler.gym.aliases:make_single_weighted_tardiness",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# 1|rj|ΣTj
register(
    id="SingleDynamicTardiness-v0",
    entry_point="cpscheduler.gym.aliases:make_single_dynamic_tardiness",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

# 1|prec|ΣCj
register(
    id="SinglePrecedenceCompletion-v0",
    entry_point="cpscheduler.gym.aliases:make_single_precedence_completion",
    max_episode_steps=None,
    disable_env_checker=True,
    order_enforce=False,
)

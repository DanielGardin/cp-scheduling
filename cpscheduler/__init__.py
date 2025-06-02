from .environment import register_env

from .common_envs import JobShopEnv, ResourceConstraintEnv

register_env(JobShopEnv, 'jobshop')
register_env(ResourceConstraintEnv, 'rcpsp')
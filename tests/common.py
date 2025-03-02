from pathlib import Path

from cpscheduler.environment import SchedulingCPEnv, JobShopSetup, Makespan
from cpscheduler.instances import read_jsp_instance

TEST_INSTANCES = [
    "dmu04",
    "la10",
    "orb01",
    "swv12",
    "ta20",
    "lta_j10_m10_1",
]

PROJECT_ROOT = Path(__file__).parent.parent

def env_setup(instance_name: str) -> SchedulingCPEnv:
    path = PROJECT_ROOT / f"instances/jobshop/{instance_name}.txt"

    instance, metadata = read_jsp_instance(path)

    env = SchedulingCPEnv(
        JobShopSetup(),
    )

    env.set_objective(Makespan())
    env.set_instance(instance, jobs='job')

    return env
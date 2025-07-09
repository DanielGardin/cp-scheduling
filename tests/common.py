from pathlib import Path

from typing import cast

from cpscheduler.environment import SchedulingEnv, JobShopSetup, Makespan
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


def env_setup(instance_name: str) -> SchedulingEnv:
    path = PROJECT_ROOT / f"instances/jobshop/{instance_name}.txt"

    try:
        instance, _ = read_jsp_instance(path)

    except FileNotFoundError as e:
        if not (path / "instances").exists():
            raise FileNotFoundError(
                f"Could not locate `instances` directory. Maybe you forgot to run `git submodule update --init`?"
            )

        raise e

    env = SchedulingEnv(
        machine_setup=JobShopSetup(),
        objective=Makespan(),
        instance_config={'instance': instance}
    )

    return env

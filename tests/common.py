from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from cpscheduler.environment import (
    FlowShopSetup,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    Makespan,
    OpenShopSetup,
    SchedulingEnv,
    SingleMachineSetup,
)
from cpscheduler.environment.constraints import PreemptionConstraint
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

EnvFactory: TypeAlias = Callable[[], SchedulingEnv]


def _single_machine_case() -> SchedulingEnv:
    return SchedulingEnv(
        machine_setup=SingleMachineSetup(disjunctive=False),
        objective=Makespan(),
        instance={"processing_time": [2, 1, 3, 2]},
        debug_mode=True,
    )


def _identical_parallel_case() -> SchedulingEnv:
    return SchedulingEnv(
        machine_setup=IdenticalParallelMachineSetup(
            n_machines=2, disjunctive=False
        ),
        objective=Makespan(),
        instance={"processing_time": [2, 1, 3, 2]},
        debug_mode=True,
    )


def _flow_shop_case() -> SchedulingEnv:
    return SchedulingEnv(
        machine_setup=FlowShopSetup(disjunctive=False),
        objective=Makespan(),
        instance={
            "job": [0, 0, 1, 1],
            "operation": [0, 1, 0, 1],
            "processing_time": [2, 3, 1, 2],
        },
        debug_mode=True,
    )


def _open_shop_case() -> SchedulingEnv:
    return SchedulingEnv(
        machine_setup=OpenShopSetup(disjunctive=False),
        objective=Makespan(),
        instance={
            "job": [0, 0, 1, 1],
            "machine": [0, 1, 0, 1],
            "processing_time": [2, 1, 3, 2],
        },
        debug_mode=True,
    )


def env_setup(
    instance_name: str, allow_preemption: bool = False
) -> SchedulingEnv:
    path = PROJECT_ROOT / f"instances/jobshop/{instance_name}.txt"

    try:
        instance, _ = read_jsp_instance(path)

    except FileNotFoundError as e:
        if not (PROJECT_ROOT / "instances").exists():
            raise FileNotFoundError(
                "Could not locate `instances` directory. Maybe you forgot to run `git submodule update --init`?"
            ) from e

        raise e

    return SchedulingEnv(
        machine_setup=JobShopSetup(),
        constraints=(PreemptionConstraint(),) if allow_preemption else (),
        objective=Makespan(),
        instance=instance,
        debug_mode=True,
    )


ENV_CASES: dict[str, EnvFactory] = {
    "jobshop_ta20": lambda: env_setup("ta20"),
    "single_machine": _single_machine_case,
    "identical_parallel": _identical_parallel_case,
    "flow_shop": _flow_shop_case,
    "open_shop": _open_shop_case,
}

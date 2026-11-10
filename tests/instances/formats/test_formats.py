from pathlib import Path
from random import Random

import pytest

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    Makespan,
    PrecedenceConstraint,
    SchedulingEnv,
    SingleMachineSetup,
    WeightedTardiness,
)
from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.heuristics.pdrs import ShortestProcessingTime
from cpscheduler.instances.formats import (
    read_rcpsp_instance,
    read_smtwt_instance,
    read_standard_jobshop_instance,
    read_taillard_jobshop_instance,
    write_smtwt_instance,
    write_standard_jobshop_instance,
    write_taillard_jobshop_instance,
)
from tests.conftest import PROJECT_ROOT
from tests.schedule_checks import (
    machine_conflicts,
    run_random_episode,
    schedule_of,
)

INSTANCES = PROJECT_ROOT / "instances"

JOB_SHOP = {
    "job": [0, 0, 1, 1],
    "operation": [0, 1, 0, 1],
    "machine": [0, 1, 1, 0],
    "processing_time": [3, 2, 2, 4],
}

needs_instances = pytest.mark.skipif(
    not INSTANCES.exists(), reason="instances submodule missing."
)


def test_standard_job_shop_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "instance.txt"

    write_standard_jobshop_instance(JOB_SHOP, path)
    instance, _ = read_standard_jobshop_instance(path)

    assert instance == JOB_SHOP


def test_taillard_job_shop_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "instance.txt"

    write_taillard_jobshop_instance(JOB_SHOP, path)
    instance, _ = read_taillard_jobshop_instance(path)

    assert instance == JOB_SHOP


def test_smtwt_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "instance.txt"
    data = {"processing_time": [1, 2], "weight": [3, 4], "due_date": [5, 6]}

    write_smtwt_instance(data, path)
    instance, metadata = read_smtwt_instance(path)

    assert instance == data
    assert metadata["n_jobs"] == 2


@needs_instances
def test_smtwt_benchmark_runs_end_to_end() -> None:
    instance, _ = read_smtwt_instance(INSTANCES / "smtwt" / "wt100_1.txt")

    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=WeightedTardiness(),
        observation=DefaultObservation(),
        instance=instance,
    )
    obs, _ = env.reset()

    *_, terminated, _, _ = env.step(
        ShortestProcessingTime().ranking(obs, "parallel")
    )

    assert terminated
    assert not machine_conflicts(schedule_of(env.state))


@needs_instances
def test_rcpsp_benchmark_precedences_can_be_scheduled() -> None:
    instance, globals_, _ = read_rcpsp_instance(
        INSTANCES / "rcpsp" / "1kAD_1.rcp"
    )
    processing_time = instance["processing_time"]

    env = SchedulingEnv(
        IdenticalParallelMachineSetup(len(processing_time), disjunctive=False),
        [PrecedenceConstraint(globals_["precedence"])],
        Makespan(),
        instance={"processing_time": processing_time},
    )
    run_random_episode(env, Random(0))

    assert env.state.remaining_tasks == 0

from random import Random

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    JobShopSetup,
    Makespan,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
    TotalTardiness,
)
from cpscheduler.instances.generators import (
    Benchmark,
    CyclicGenerator,
    Generator,
)
from tests.schedule_checks import (
    machine_conflicts,
    run_random_episode,
    schedule_of,
)


def test_taillard_generator_samples_rectangular_job_shops() -> None:
    env = SchedulingEnv(JobShopSetup(), objective=Makespan())
    generator = Benchmark.create(
        env, "taillard", n_tasks=12, n_machines=3, n_jobs=4, seed=0
    )

    sample = generator.sample(0)

    assert sample == generator.sample(0)
    assert sample["job"] == [j for j in range(4) for _ in range(3)]
    assert all(1 <= p <= 99 for p in sample["processing_time"])
    assert all(
        sorted(sample["machine"][3 * job : 3 * job + 3]) == [0, 1, 2]
        for job in range(4)
    )

    env.set_generator(generator)
    run_random_episode(env, Random(0))

    assert env.state.remaining_tasks == 0
    assert not machine_conflicts(schedule_of(env.state))


def test_generators_with_the_same_seed_sample_the_same_sequence() -> None:
    env = SchedulingEnv(SingleMachineSetup(), objective=Makespan())

    first = Generator.from_env(env, n_tasks=5, seed=1)
    second = Generator.from_env(env, n_tasks=5, seed=1)

    a, b = first.sample(), first.sample()

    assert a != b
    assert a == second.sample()


def test_default_samplers_cover_the_required_features() -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [ReleaseDateConstraint()],
        TotalTardiness(),
    )

    env.load_instance(Generator.from_env(env, n_tasks=5, seed=1).sample())
    run_random_episode(env, Random(0))

    assert env.state.remaining_tasks == 0


def test_cyclic_generator_changes_the_instance_on_reset() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        instance=CyclicGenerator(
            [{"processing_time": [1]}, {"processing_time": [1, 2]}]
        ),
    )

    n_tasks = []
    for _ in range(3):
        env.reset()
        n_tasks.append(env.state.n_tasks)

    assert n_tasks == [1, 2, 1]

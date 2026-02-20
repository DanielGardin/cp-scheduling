# from cpscheduler.environment.env import SchedulingEnv
# from cpscheduler.environment.constraints import DisjunctiveConstraint, MachineConstraint, PrecedenceConstraint
# from cpscheduler.environment.objectives import Makespan
# from cpscheduler.environment.schedule_setup import (
#     SingleMachineSetup,
#     IdenticalParallelMachineSetup,
#     UniformParallelMachineSetup,
#     UnrelatedParallelMachineSetup,
#     OpenShopSetup,
#     JobShopSetup,
# )
# from cpscheduler.instances.common import generate_instance

# def test_single_machine_setup_assigns_machine_zero() -> None:
#     instance = {"processing_time": [3, 2, 5]}
#     env = _make_env(instance, SingleMachineSetup(disjunctive=False))

#     for task in env.state.tasks:
#         assert list(task.machines) == [0]


# def test_identical_parallel_machine_setup_assigns_all_machines() -> None:
#     instance = {"processing_time": [4, 2]}
#     env = _make_env(instance, IdenticalParallelMachineSetup(n_machines=3, disjunctive=False))

#     for task, p_time in zip(env.state.tasks, instance["processing_time"]):
#         assert set(task.machines) == {0, 1, 2}
#         assert all(value == p_time for value in task.processing_times.values())


# def test_uniform_parallel_machine_setup_scales_processing_times() -> None:
#     instance = {"processing_time": [3]}
#     env = _make_env(instance, UniformParallelMachineSetup(speed=[1, 2], disjunctive=False))

#     task = env.state.tasks[0]
#     assert task.processing_times[0] == 3
#     assert task.processing_times[1] == 2


# def test_unrelated_parallel_machine_setup_uses_feature_times() -> None:
#     instance = {"p0": [3, 5], "p1": [2, 7]}
#     env = _make_env(instance, UnrelatedParallelMachineSetup(["p0", "p1"], disjunctive=False))

#     assert env.state.tasks[0].processing_times[0] == 3
#     assert env.state.tasks[0].processing_times[1] == 2
#     assert env.state.tasks[1].processing_times[0] == 5
#     assert env.state.tasks[1].processing_times[1] == 7


# def test_open_shop_setup_builds_disjunctive_constraints() -> None:
#     instance = {
#         "job": [0, 0, 1, 1],
#         "p0": [3, 2, 4, 1],
#         "p1": [1, 2, 1, 3],
#     }
#     env = _make_env(instance, OpenShopSetup(["p0", "p1"], disjunctive=True))

#     assert any(isinstance(c, DisjunctiveConstraint) for c in env.setup_constraints)
#     assert any(isinstance(c, MachineConstraint) for c in env.setup_constraints)


# def test_job_shop_setup_builds_machine_and_precedence_constraints() -> None:
#     instance, _ = generate_instance(
#         n_jobs=2,
#         n_machines=2,
#         processing_time_dist=lambda: 3,
#         setup="jobshop",
#         seed=7,
#     )
#     env = _make_env(instance, JobShopSetup())

#     assert any(isinstance(c, MachineConstraint) for c in env.setup_constraints)
#     assert any(isinstance(c, PrecedenceConstraint) for c in env.setup_constraints)
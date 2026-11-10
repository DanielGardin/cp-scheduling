from collections.abc import Callable
from random import Random
from typing import NoReturn

import pytest

pytest.importorskip("pyomo")
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory

from cpscheduler.environment import (
    AtMostOneConstraint,
    ComposedObjective,
    ExactlyOneConstraint,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    Makespan,
    Objective,
    OptionalityConstraint,
    PrecedenceConstraint,
    RejectableConstraint,
    RejectionCost,
    ReleaseDateConstraint,
    SchedulingEnv,
    SetupConstraint,
    SingleMachineSetup,
    TotalCompletionTime,
    TotalEarliness,
    TotalTardiness,
    WeightedCompletionTime,
    WeightedEarliness,
)
from cpscheduler.solver import SchedulingSolver
from cpscheduler.solver.milp.disjunctive.formulation import (
    DisjunctiveMILPFormulation,
)
from cpscheduler.solver.milp.pyomo_formulation import PYOMO_PARAM
from tests.schedule_checks import random_job_shop
from tests.solver_checks import (
    REGULAR_OBJECTIVES,
    InstanceData,
    brute_force,
    regular_instance,
    replay,
)

MILP_SOLVER = next(
    (
        tag
        for tag in ("cbc", "glpk", "appsi_highs", "gurobi")
        if SolverFactory(tag).available(exception_flag=False)
    ),
    "",
)

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not MILP_SOLVER, reason="No MILP solver available."),
]


def _solve_and_replay(env: SchedulingEnv) -> tuple[float, float]:
    env.reset()
    action, value, _ = SchedulingSolver(
        env, DisjunctiveMILPFormulation()
    ).solve(solver_tag=MILP_SOLVER, quiet=True, time_limit=30)

    return value, replay(env, action)


@pytest.mark.parametrize("objective", REGULAR_OBJECTIVES)
def test_milp_is_optimal_with_parallel_machines(objective: str) -> None:
    # Includes objectives whose optimum finishes after a greedy schedule
    factory, features = REGULAR_OBJECTIVES[objective]
    rng = Random(5)

    for _ in range(3):
        instance = regular_instance(rng, 4, features)

        def make(
            backend: str = "des", instance: InstanceData = instance
        ) -> SchedulingEnv:
            return SchedulingEnv(
                IdenticalParallelMachineSetup(2),
                objective=factory(),
                backend=backend,
                instance=instance,
            )

        value, replayed = _solve_and_replay(make())

        assert value == pytest.approx(brute_force(lambda: make("tetris"), 4))
        assert replayed == pytest.approx(value)


@pytest.mark.parametrize("objective", [*REGULAR_OBJECTIVES, "earliness"])
def test_milp_is_optimal_with_optional_tasks(objective: str) -> None:
    # A rejected job contributes nothing, so rejecting may pay off
    factory, features = (
        (TotalEarliness, ["due_date"])
        if objective == "earliness"
        else REGULAR_OBJECTIVES[objective]
    )
    rng = Random(8)

    for _ in range(3):
        instance = regular_instance(rng, 4, features)

        def make(
            backend: str = "des", instance: InstanceData = instance
        ) -> SchedulingEnv:
            return SchedulingEnv(
                IdenticalParallelMachineSetup(2),
                [OptionalityConstraint()],
                factory(),
                backend=backend,
                instance=instance,
            )

        value, replayed = _solve_and_replay(make())

        assert value == pytest.approx(
            brute_force(lambda: make("tetris"), 4, rejectable=True)
        )
        assert replayed == pytest.approx(value)


@pytest.mark.parametrize("objective", REGULAR_OBJECTIVES)
def test_milp_is_optimal_with_rejection_costs(objective: str) -> None:
    # Rejecting saves the cost of scheduling a task, at a price
    factory, features = REGULAR_OBJECTIVES[objective]
    rng = Random(10)

    for _ in range(3):
        instance = regular_instance(rng, 4, features)
        costs = [float(rng.randint(0, 12)) for _ in range(4)]

        def make(
            backend: str = "des",
            instance: InstanceData = instance,
            costs: list[float] = costs,
        ) -> SchedulingEnv:
            return SchedulingEnv(
                IdenticalParallelMachineSetup(2),
                [OptionalityConstraint()],
                ComposedObjective(
                    [factory(), RejectionCost(rejection_costs=costs)]
                ),
                backend=backend,
                instance=instance,
            )

        value, replayed = _solve_and_replay(make())

        assert value == pytest.approx(
            brute_force(lambda: make("tetris"), 4, rejectable=True)
        )
        assert replayed == pytest.approx(value)


@pytest.mark.parametrize(
    "objective",
    [Makespan, TotalCompletionTime, WeightedCompletionTime, TotalTardiness],
)
def test_milp_is_optimal_with_rejectable_jobs(
    objective: Callable[[], Objective],
) -> None:
    # A job is kept or rejected as a whole, at the price of its rejection cost
    rng = Random(12)

    for _ in range(3):
        instance: InstanceData = {
            "job": [0, 0, 1, 1],
            "processing_time": [rng.randint(1, 9) for _ in range(4)],
            "weight": [float(rng.randint(1, 3)) for _ in range(2)],
            "due_date": [rng.randint(3, 12) for _ in range(2)],
            "rejection_cost": [float(rng.randint(0, 25)) for _ in range(2)],
        }

        def make(
            backend: str = "des", instance: InstanceData = instance
        ) -> SchedulingEnv:
            env = SchedulingEnv(
                IdenticalParallelMachineSetup(2),
                [RejectableConstraint()],
                ComposedObjective([objective(), RejectionCost()]),
                backend=backend,
            )
            features = set(env.required_features()) | {"job", "processing_time"}
            env.load_instance(
                {k: v for k, v in instance.items() if k in features}
            )

            return env

        value, replayed = _solve_and_replay(make())

        assert value == pytest.approx(
            brute_force(lambda: make("tetris"), 4, rejectable=True)
        )
        assert replayed == pytest.approx(value)


@pytest.mark.parametrize("group", [ExactlyOneConstraint, AtMostOneConstraint])
@pytest.mark.parametrize("objective", REGULAR_OBJECTIVES)
def test_milp_is_optimal_with_alternative_tasks(
    objective: str, group: type[AtMostOneConstraint]
) -> None:
    # With exactly one task of each pair, rejecting every task is not an option
    factory, features = REGULAR_OBJECTIVES[objective]
    rng = Random(9)

    for _ in range(3):
        instance = regular_instance(rng, 4, features)

        def make(
            backend: str = "des", instance: InstanceData = instance
        ) -> SchedulingEnv:
            return SchedulingEnv(
                IdenticalParallelMachineSetup(2),
                [OptionalityConstraint(), group([[0, 1], [2, 3]])],
                factory(),
                backend=backend,
                instance=instance,
            )

        value, replayed = _solve_and_replay(make())

        assert value == pytest.approx(
            brute_force(lambda: make("tetris"), 4, rejectable=True)
        )
        assert replayed == pytest.approx(value)


def test_milp_starts_from_tasks_already_rejected() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [OptionalityConstraint()],
        TotalCompletionTime(),
        instance={"processing_time": [3, 1, 2]},
    )
    env.reset()
    env.step(("reject", 1))

    action, value, _ = SchedulingSolver(
        env, DisjunctiveMILPFormulation()
    ).solve(solver_tag=MILP_SOLVER, quiet=True)

    assert value == 0
    assert ("reject", 1) not in action


def test_milp_warm_start_may_reject_tasks() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [OptionalityConstraint()],
        Makespan(),
        instance={"processing_time": [3, 1]},
    )
    env.reset()

    solver = SchedulingSolver(env, DisjunctiveMILPFormulation())
    solver.warm_start([("reject", 0), ("execute", 1)])
    action, value, _ = solver.solve(solver_tag=MILP_SOLVER, quiet=True)

    assert solver.found_solution
    assert value == replay(env, action) == 0


@pytest.mark.parametrize("factory", [TotalEarliness, WeightedEarliness])
def test_milp_earliness_is_the_value_of_its_schedule(
    factory: Callable[[], Objective],
) -> None:
    # Earliness improves with later completions, it needs exact variables.
    # [0, 3) and [7, 9) are on time.
    instance: InstanceData = {"processing_time": [3, 2], "due_date": [3, 9]}

    if factory is WeightedEarliness:
        instance["weight"] = [1.0, 1.0]

    env = SchedulingEnv(
        SingleMachineSetup(), objective=factory(), instance=instance
    )
    value, replayed = _solve_and_replay(env)

    assert value == pytest.approx(0)
    assert replayed == pytest.approx(value)


def test_milp_composed_objective() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=ComposedObjective(
            [Makespan(), TotalCompletionTime()], [1.0, 2.0]
        ),
        instance={"processing_time": [3, 1, 2]},
    )
    value, replayed = _solve_and_replay(env)

    assert value == pytest.approx(6 + 2 * (1 + 3 + 6))
    assert replayed == pytest.approx(value)


def test_milp_setup_times_apply_in_both_orders() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [SetupConstraint({0: {1: 5}, 1: {0: 1}})],
        Makespan(),
        instance={"processing_time": [1, 1]},
    )
    value, replayed = _solve_and_replay(env)

    # Task 1, then the setup from 1 to 0, then task 0
    assert value == pytest.approx(3)
    assert replayed == pytest.approx(value)


def test_milp_setup_times_ignore_other_machines() -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [SetupConstraint({0: {1: 5}, 1: {0: 5}})],
        Makespan(),
        instance={"processing_time": [1, 1]},
    )
    value, replayed = _solve_and_replay(env)

    assert value == pytest.approx(1)
    assert replayed == pytest.approx(value)


def test_milp_respects_release_dates_and_precedence() -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [ReleaseDateConstraint(), PrecedenceConstraint({1: [0], 2: [1]})],
        Makespan(),
        instance={"processing_time": [2, 2, 2], "release_time": [4, 0, 0]},
    )
    value, replayed = _solve_and_replay(env)

    assert value == pytest.approx(10)
    assert replayed == pytest.approx(value)


def _large_job_shop() -> SchedulingEnv:
    # Too large for the solver to find a schedule within a second
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        instance=random_job_shop(Random(0), 50, 10),
    )
    env.reset()

    return env


def test_milp_without_a_schedule_is_an_error() -> None:
    solver = SchedulingSolver(_large_job_shop(), DisjunctiveMILPFormulation())

    with pytest.raises(RuntimeError, match="did not find a solution"):
        solver.solve(solver_tag=MILP_SOLVER, quiet=True, time_limit=1)

    assert not solver.found_solution


def test_milp_without_a_schedule_returns_the_warm_start() -> None:
    env = _large_job_shop()
    warm_start = [("submit", task) for task in range(env.state.n_tasks)]
    warm_value = replay(env, warm_start)

    env.reset()
    solver = SchedulingSolver(env, DisjunctiveMILPFormulation())
    solver.warm_start(warm_start)

    action, value, _ = solver.solve(
        solver_tag=MILP_SOLVER, quiet=True, time_limit=1
    )

    assert value <= warm_value
    assert replay(env, action) == pytest.approx(value)

    if not solver.found_solution:
        assert value == warm_value


def test_milp_horizon_only_bounds_present_tasks() -> None:
    # Task 0 cannot finish by the horizon, it must be rejected instead
    env = SchedulingEnv(
        SingleMachineSetup(),
        [OptionalityConstraint()],
        TotalTardiness(due_dates=[0, 0]),
        instance={"processing_time": [5, 1]},
    )
    env.reset()

    action, value, _ = SchedulingSolver(
        env, DisjunctiveMILPFormulation(horizon=2)
    ).solve(solver_tag=MILP_SOLVER, quiet=True)

    assert ("reject", 0) in action
    assert replay(env, action) == pytest.approx(value)


class _CrashingSolver:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.options: dict[str, object] = {}

    def warm_start_capable(self) -> bool:
        return True

    def solve(self, *args: object, **kwargs: object) -> NoReturn:
        raise ApplicationError("Solver (cbc) did not exit normally")


def test_milp_crashed_solver_returns_the_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CBC can crash at a time limit, the process produced no schedule
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        instance={"processing_time": [3, 1]},
    )
    env.reset()

    solver = SchedulingSolver(env, DisjunctiveMILPFormulation())
    solver.warm_start([("execute", 1), ("execute", 0)])
    monkeypatch.setattr(
        "cpscheduler.solver.milp.pyomo_formulation.pyo.SolverFactory",
        _CrashingSolver,
    )

    action, value, _ = solver.solve(solver_tag="cbc", quiet=True)

    assert not solver.found_solution
    assert value == replay(env, action) == 4


def _model_with_an_absent_literal() -> tuple[
    DisjunctiveMILPFormulation, PYOMO_PARAM
]:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        instance={"processing_time": [1]},
    )
    env.reset()

    formulation = DisjunctiveMILPFormulation()
    formulation.initialize_model(env)

    absent = formulation.add_var("absent", binary=True)
    formulation.add_constraint(absent == 0, "absent_is_off")

    return formulation, absent


def test_exact_conditional_max_ignores_values_without_their_condition() -> None:
    # Otherwise an absent task, whose end is free, can be picked as the latest
    formulation, absent = _model_with_an_absent_literal()
    late = formulation.add_var("late", lb=0, ub=10)

    maximum = formulation.max_expr(
        [3, late], "maximum", exact=True, conditions=[1, absent]
    )
    formulation.set_objective(-maximum)
    formulation.solve(solver_tag=MILP_SOLVER, quiet=True)

    assert formulation.get_value(maximum) == pytest.approx(3)


def test_conditional_max_without_any_condition_is_the_default() -> None:
    formulation, absent = _model_with_an_absent_literal()
    early = formulation.add_var("early", lb=-10, ub=10)

    maximum = formulation.max_expr(
        [early], "maximum", conditions=[absent], default=0
    )
    formulation.set_objective(maximum)
    formulation.solve(solver_tag=MILP_SOLVER, quiet=True)

    assert formulation.get_value(maximum) == pytest.approx(0)

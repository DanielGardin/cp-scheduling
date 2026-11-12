from collections.abc import Callable
from math import exp
from random import Random
from typing import TYPE_CHECKING, Literal, TypeAlias

import pytest

from cpscheduler.environment import (
    AntiRegularObjective,
    AtMostOneConstraint,
    ComposedObjective,
    DeadlineConstraint,
    DiscountedTotalCompletionTime,
    ExactlyOneConstraint,
    HorizonConstraint,
    IdenticalParallelMachineSetup,
    JobShopSetup,
    Makespan,
    MaximumLateness,
    NoWaitConstraint,
    Objective,
    OptionalityConstraint,
    PrecedenceConstraint,
    RegularObjective,
    RejectableConstraint,
    RejectionCost,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
    TotalCompletionTime,
    TotalEarliness,
    TotalFlowTime,
    TotalTardiness,
    TotalTardyJobs,
    UnrelatedParallelMachineSetup,
    WeightedCompletionTime,
    WeightedEarliness,
    WeightedTardiness,
    WeightedTardyJobs,
)
from cpscheduler.environment.objectives import SatisfactionObjective
from cpscheduler.environment.objectives.base import MaxCompletionObjective
from cpscheduler.environment.state import ScheduleState

if TYPE_CHECKING:
    from cpscheduler.environment.constants import Int
    from cpscheduler.environment.transition import ActionType

CompletionObjective: TypeAlias = (
    RegularObjective | AntiRegularObjective | MaxCompletionObjective
)
Bounds: TypeAlias = tuple[float, float, float]
"(lb, ub, value)"

# Tasks run in order: 0 in [0, 3) and 1 in [3, 5), so C = (3, 5)
DATA: dict[str, list[int] | list[float]] = {
    "due_date": [4, 1],
    "weight": [2.0, 1.0],
    "release_time": [0, 1],
}

OBJECTIVES = [
    pytest.param(Makespan, [], 5, id="makespan"),
    pytest.param(TotalCompletionTime, [], 3 + 5, id="completion"),
    pytest.param(
        WeightedCompletionTime, ["weight"], 2 * 3 + 5, id="weighted_completion"
    ),
    pytest.param(TotalFlowTime, ["release_time"], 3 + 4, id="flow_time"),
    pytest.param(MaximumLateness, ["due_date"], 4, id="max_lateness"),
    pytest.param(TotalTardiness, ["due_date"], 4, id="tardiness"),
    pytest.param(
        WeightedTardiness, ["due_date", "weight"], 4, id="weighted_tardiness"
    ),
    pytest.param(TotalEarliness, ["due_date"], 1, id="earliness"),
    pytest.param(
        WeightedEarliness, ["due_date", "weight"], 2, id="weighted_earliness"
    ),
    pytest.param(TotalTardyJobs, ["due_date"], 1, id="tardy_jobs"),
    pytest.param(
        WeightedTardyJobs, ["due_date", "weight"], 1, id="weighted_tardy_jobs"
    ),
    pytest.param(
        lambda: DiscountedTotalCompletionTime(0.5),
        [],
        (1 - exp(-0.5 * 3)) + (1 - exp(-0.5 * 5)),
        id="discounted",
    ),
]


@pytest.mark.parametrize(("factory", "features", "expected"), OBJECTIVES)
def test_objective_matches_its_definition(
    factory: Callable[[], Objective], features: list[str], expected: float
) -> None:
    instance = {
        "processing_time": [3, 2],
        **{key: DATA[key] for key in features},
    }
    env = SchedulingEnv(
        SingleMachineSetup(), objective=factory(), instance=instance
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step(
        [("execute", 0), ("execute", 1)]
    )

    assert (terminated, truncated) == (True, False)
    assert env.objective.value == pytest.approx(expected)
    assert env.objective.compute(env.state) == pytest.approx(expected)
    assert info["objective_value"] == pytest.approx(expected)
    assert reward == pytest.approx(-expected)


@pytest.mark.parametrize(
    "objective",
    [
        TotalTardiness(due_dates=[1]),
        WeightedTardiness(due_dates=[1], weights=[2.0]),
        TotalEarliness(due_dates=[1]),
        WeightedEarliness(due_dates=[1], weights=[2.0]),
        TotalTardyJobs(due_dates=[1]),
        WeightedTardyJobs(due_dates=[1], weights=[2.0]),
        MaximumLateness(due_dates=[1]),
    ],
    ids=lambda objective: type(objective).__name__,
)
def test_constructor_data_replaces_instance_data(objective: Objective) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=objective,
        instance={"processing_time": [3]},
    )
    env.reset()

    env.step(("execute", 0))

    assert env.state.is_terminal()


def test_maximum_lateness_can_be_negative() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=MaximumLateness(due_dates=[10, 10]),
        instance={"processing_time": [1, 1]},
    )
    env.reset()

    env.step([("execute", 0), ("execute", 1)])

    assert env.objective.value == -8


def test_composed_objective_is_the_linear_combination() -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=ComposedObjective(
            [Makespan(), WeightedTardiness()], coefficients=[1.0, 2.0]
        ),
        instance={
            "job": [0, 0, 1, 1],
            "operation": [0, 1, 0, 1],
            "machine": [0, 1, 1, 0],
            "processing_time": [3, 2, 2, 4],
            "due_date": [1, 1],
            "weight": [1.0, 3.0],
        },
    )
    env.reset()

    env.step([("execute", i) for i in (0, 2, 1, 3)])

    # C = (5, 7): C_max = 7 and Σw_jT_j = 1 * 4 + 3 * 6
    assert env.objective.value == 7 + 2 * (4 + 18)


def test_lower_bound_includes_unscheduled_work_on_reset() -> None:
    env = SchedulingEnv(
        JobShopSetup(),
        objective=Makespan(),
        instance={
            "job": [0, 0],
            "operation": [0, 1],
            "machine": [0, 1],
            "processing_time": [3, 4],
        },
    )
    env.reset()

    assert env.objective.lb == 7


@pytest.mark.parametrize(
    ("objective", "kept_cost"),
    [
        # Job 1 is rejected and job 0 runs alone in [0, 2)
        (TotalFlowTime(release_times=[0, 4]), 2.0),
        (TotalEarliness(due_dates=[5, 9]), 3.0),
        (TotalCompletionTime(), 2.0),
        (MaximumLateness(due_dates=[1, 0]), 1.0),
    ],
    ids=["flow_time", "earliness", "completion", "max_lateness"],
)
def test_rejected_jobs_cost_nothing(
    objective: Objective, kept_cost: float
) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [OptionalityConstraint()],
        objective,
        instance={"processing_time": [2, 3]},
    )
    env.reset()

    env.step([("reject", 1), ("execute", 0)])

    assert env.state.remaining_tasks == 0
    assert objective.value == objective.compute(env.state) == kept_cost
    assert objective.lb == objective.ub == kept_cost


@pytest.mark.parametrize(
    ("processing", "horizon", "actions", "expected"),
    [
        pytest.param([2], None, [("execute", 0)], (1, 1, 1), id="one_task"),
        pytest.param(
            [2, 2],
            None,
            [("execute", 0), ("execute", 1)],
            (1, 1, 1),
            id="two_steps",
        ),
        pytest.param(
            [2, 2],
            None,
            [[("execute", 0), ("execute", 1)]],
            (1, 1, 1),
            id="batched",
        ),
        pytest.param([2, 2], 3, [("execute", 0)], (0, 0, 0), id="fails_first"),
        pytest.param(
            [2, 2, 2],
            5,
            [("execute", 0), ("execute", 1)],
            (0, 0, 0),
            id="fails_later",
        ),
    ],
)
def test_satisfaction_scores_only_complete_feasible_schedules(
    processing: list[int],
    horizon: int | None,
    actions: list["ActionType"],
    expected: Bounds,
) -> None:
    objective = SatisfactionObjective()
    env = SchedulingEnv(
        SingleMachineSetup(),
        [HorizonConstraint(horizon)] if horizon is not None else [],
        objective,
        instance={"processing_time": processing},
    )
    env.reset()

    for action in actions:
        assert (objective.lb, objective.ub, objective.value) == (0, 1, 0)

        env.step(action)

    assert env.state.is_terminal()
    assert (objective.lb, objective.ub, objective.value) == expected
    assert objective.value == objective.compute(env.state)

    env.reset()

    assert (objective.lb, objective.ub, objective.value) == (0, 1, 0)


def test_every_term_of_a_composition_is_told_of_the_failure() -> None:
    failures: list[bool] = []

    # Compiled objectives only accept interpreted subclasses of the base class
    class Recorder(Objective):
        @property
        def value(self) -> float:
            return 0.0

        def on_failure(self, state: ScheduleState) -> None:
            failures.append(state.infeasible)

    env = SchedulingEnv(
        SingleMachineSetup(),
        [HorizonConstraint(3)],
        ComposedObjective([Recorder(), Recorder()]),
        instance={"processing_time": [2, 2]},
    )
    env.reset()

    env.step(("execute", 0))

    assert env.state.infeasible
    assert failures == [True, True]


def _reference_bounds(
    objective: CompletionObjective, state: ScheduleState
) -> Bounds:
    # A present job completes between the end of its present tasks and the end
    # of the tasks that are not absent. A job without present tasks may still be
    # rejected, costing 0, and a rejected one always costs 0.
    n_jobs = state.n_jobs
    lbs: list[int | None] = [None] * n_jobs
    ubs: list[int | None] = [None] * n_jobs
    partial: list[int | None] = [None] * n_jobs

    def _max(current: int | None, new: int) -> int:
        return new if current is None else max(current, new)

    for task in range(state.n_tasks):
        if state.is_absent(task):
            continue

        job = state.get_job_id(task)
        ubs[job] = _max(ubs[job], state.get_end_ub(task))

        if state.is_present(task):
            lbs[job] = _max(lbs[job], state.get_end_lb(task))

        if state.is_assigned(task):
            partial[job] = _max(partial[job], state.get_end(task))

    is_max = isinstance(objective, MaxCompletionObjective)
    increasing = (
        True
        if isinstance(objective, MaxCompletionObjective)
        else objective.increasing
    )
    cost = objective.job_cost

    low: list[float] = []
    high: list[float] = []
    for job, (lb, ub) in enumerate(zip(lbs, ubs, strict=True)):
        if ub is None:
            continue

        if lb is not None:
            early, late = (lb, ub) if increasing else (ub, lb)
            low.append(cost(job, early))
            high.append(cost(job, late))

        elif is_max:
            # A maximum leaves a job that may be rejected out of its lower bound
            high.append(cost(job, ub))

        else:
            costs = (cost(job, 0), cost(job, ub))
            low.append(min(0.0, *costs))
            high.append(max(0.0, *costs))

    value = [cost(job, c) for job, c in enumerate(partial) if c is not None]

    def aggregate(costs: list[float]) -> float:
        return max(costs, default=0.0) if is_max else sum(costs)

    return aggregate(low), aggregate(high), aggregate(value)


INCREMENTAL: dict[str, Callable[[], CompletionObjective]] = {
    "makespan": Makespan,
    "max_lateness": lambda: MaximumLateness(due_dates=[5] * 6),
    "completion": TotalCompletionTime,
    "weighted_completion": lambda: WeightedCompletionTime(
        weights=[0.5, 2.0] * 3
    ),
    "flow_time": TotalFlowTime,
    "tardiness": lambda: TotalTardiness(due_dates=[6] * 6),
    "tardy_jobs": lambda: TotalTardyJobs(due_dates=[6] * 6),
    "earliness": lambda: TotalEarliness(due_dates=[30] * 6),
    "discounted": lambda: DiscountedTotalCompletionTime(0.1),
}


def _incremental_env(
    objective: Objective,
    transition: str,
    rng: Random,
    links: Literal["precedence", "no_wait"] = "precedence",
) -> SchedulingEnv:
    # Release dates move lower bounds, deadlines move upper bounds, optional
    # tasks become absent or present, and both remove machines without bound
    # events of their own. No-wait links tighten start upper bounds by
    # themselves. Parents belong to another job, so each job is only refreshed
    # by the events of its own tasks.
    n_tasks = 12
    chains: dict[Int, list[Int]] = {t: [t - 5] for t in range(6, n_tasks)}

    return SchedulingEnv(
        UnrelatedParallelMachineSetup(),
        [
            ReleaseDateConstraint(),
            DeadlineConstraint(),
            OptionalityConstraint(),
            NoWaitConstraint(chains)
            if links == "no_wait"
            else PrecedenceConstraint(chains),
        ],
        objective,
        transition=transition,
        instance={
            "job": [t % 6 for t in range(n_tasks)],
            "processing_time": [
                [rng.randint(1, 9), rng.randint(1, 9)] for _ in range(n_tasks)
            ],
            "release_time": [rng.randint(0, 8) for _ in range(n_tasks)],
            "due_time": [rng.randint(15, 60) for _ in range(n_tasks)],
        },
    )


@pytest.mark.env
@pytest.mark.parametrize("links", ["precedence", "no_wait"])
@pytest.mark.parametrize("transition", ["des", "tetris"])
@pytest.mark.parametrize("objective", INCREMENTAL)
def test_incremental_bounds_match_a_computation_from_scratch(
    objective: str, transition: str, links: Literal["precedence", "no_wait"]
) -> None:
    rng = Random(11)

    for _ in range(25):
        tracked = INCREMENTAL[objective]()
        env = _incremental_env(tracked, transition, rng, links)
        env.reset()

        checkpoints: list[int] = []
        for _ in range(60):
            # Bounds are meaningless in an infeasible state, which ends the episode
            if env.state.infeasible:
                break

            assert (tracked.lb, tracked.ub, tracked.value) == pytest.approx(
                _reference_bounds(tracked, env.state)
            )

            if env.state.is_terminal():
                break

            choice = rng.random()
            support = env.get_action_support()

            if choice < 0.15:
                checkpoints.append(env.checkpoint())

            elif choice < 0.25 and checkpoints:
                env.backtrack(checkpoints.pop())

            elif choice < 0.4 and support:
                env.step(("reject", rng.choice(support)))

            elif choice < 0.5 and support:
                env.state.require_task(rng.choice(support))
                env.propagate()

            else:
                env.step(("execute", rng.choice(support)) if support else None)

        if env.state.remaining_tasks == 0 and not env.state.infeasible:
            assert tracked.value == pytest.approx(tracked.compute(env.state))


BOUNDED: dict[str, Callable[[], Objective]] = {
    **INCREMENTAL,
    "composed": lambda: ComposedObjective(
        [Makespan(), TotalCompletionTime()], [1.0, 2.0]
    ),
}


@pytest.mark.env
@pytest.mark.parametrize("objective", BOUNDED)
def test_bounds_contain_the_value_of_the_schedule_reached(
    objective: str,
) -> None:
    # Unlike the reference, this checks the bounds are valid: a search that
    # prunes with a wrong bound silently loses optima.
    rng = Random(5)

    for _ in range(40):
        tracked = BOUNDED[objective]()
        env = _incremental_env(tracked, "des", rng)
        env.reset()

        bounds: list[tuple[float, float]] = []
        while not env.state.is_terminal():
            bounds.append((tracked.lb, tracked.ub))
            support = env.get_action_support()
            if not support:
                break

            kind = "reject" if rng.random() < 0.2 else "execute"
            env.step((kind, rng.choice(support)))

        if env.state.infeasible or env.state.remaining_tasks:
            continue

        final = tracked.compute(env.state)

        assert tracked.value == pytest.approx(final)
        assert (tracked.lb, tracked.ub) == pytest.approx((final, final))

        for lb, ub in bounds:
            assert lb - 1e-9 <= final <= ub + 1e-9


@pytest.mark.parametrize(
    ("terms", "coefficients", "regular", "anti_regular"),
    [
        pytest.param([Makespan()], [1.0], True, False, id="regular"),
        pytest.param(
            [TotalEarliness()], [-1.0], True, False, id="negated_anti"
        ),
        pytest.param([TotalEarliness()], [1.0], False, True, id="anti"),
        pytest.param([Makespan()], [-1.0], False, True, id="negated_regular"),
        pytest.param(
            [Makespan(), TotalEarliness()], [1.0, 1.0], False, False, id="mixed"
        ),
        pytest.param(
            [Makespan(), TotalEarliness()], [2.0, 0.0], True, False, id="zero"
        ),
    ],
)
def test_coefficients_carry_the_direction_of_each_term(
    terms: list[Objective],
    coefficients: list[float],
    regular: bool,
    anti_regular: bool,
) -> None:
    composed = ComposedObjective(terms, coefficients)

    assert (composed.regular, composed.anti_regular) == (regular, anti_regular)


def test_composition_with_an_unsafe_term_cannot_checkpoint() -> None:
    class Unsafe(Objective):
        @property
        def value(self) -> float:
            return 0.0

    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=ComposedObjective([Makespan(), Unsafe()]),
        instance={"processing_time": [1]},
    )
    env.reset()

    with pytest.raises(RuntimeError, match="backtrack-unsafe"):
        env.checkpoint()


def test_rejection_cost_bounds_follow_the_decisions() -> None:
    objective = RejectionCost(rejection_costs=[2.0, 3.0, 5.0])
    env = SchedulingEnv(
        SingleMachineSetup(),
        [OptionalityConstraint()],
        objective,
        instance={"processing_time": [1, 1, 1]},
    )
    env.reset()

    assert (objective.lb, objective.ub, objective.value) == (0, 10, 0)

    env.step(("reject", 0))

    assert (objective.lb, objective.ub, objective.value) == (2, 10, 2)

    env.step(("execute", 1))

    assert (objective.lb, objective.ub, objective.value) == (2, 7, 2)

    env.step(("reject", 2))

    assert (objective.lb, objective.ub, objective.value) == (7, 7, 7)
    assert objective.compute(env.state) == 7

    env.reset()

    assert (objective.lb, objective.ub, objective.value) == (0, 10, 0)


def test_rejections_made_while_resetting_count_once() -> None:
    # Task 0 is required by its group, which rejects task 1 in the other group
    objective = RejectionCost(rejection_costs=[2.0, 3.0, 5.0])
    env = SchedulingEnv(
        SingleMachineSetup(),
        [
            OptionalityConstraint(),
            ExactlyOneConstraint([[0]]),
            AtMostOneConstraint([[0, 1]]),
        ],
        objective,
        instance={"processing_time": [1, 1, 1]},
    )
    env.reset()

    assert env.state.is_present(0)
    assert env.state.is_absent(1)
    assert (objective.lb, objective.ub, objective.value) == (3, 8, 3)


def test_partly_rejected_jobs_cost_nothing() -> None:
    objective = RejectionCost(rejection_costs=[4.0, 6.0])
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [OptionalityConstraint()],
        objective,
        instance={"job": [0, 0, 1, 1], "processing_time": [1, 1, 1, 1]},
    )
    env.reset()

    env.step(("reject", 0))

    assert (objective.lb, objective.ub, objective.value) == (0, 10, 0)

    env.step(("reject", 1))

    assert (objective.lb, objective.ub, objective.value) == (4, 10, 4)

    env.step([("execute", 2), ("reject", 3)])

    assert (objective.lb, objective.ub, objective.value) == (4, 4, 4)
    assert objective.compute(env.state) == 4


def test_rejecting_a_task_of_a_rejectable_job_charges_the_job_once() -> None:
    objective = RejectionCost(rejection_costs=[4.0, 6.0])
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(2),
        [RejectableConstraint()],
        objective,
        instance={"job": [0, 0, 1, 1], "processing_time": [1, 1, 1, 1]},
    )
    env.reset()

    env.step(("reject", 1))

    assert (objective.lb, objective.ub, objective.value) == (4, 10, 4)


def _reference_rejection(costs: list[float], state: ScheduleState) -> Bounds:
    # A job is rejected with all its tasks absent, and may still be rejected
    # while none of its tasks is present
    rejected = rejectable = 0.0
    for job, tasks in enumerate(state.instance.job_tasks):
        if all(state.is_absent(task) for task in tasks):
            rejected += costs[job]

        elif not any(state.is_present(task) for task in tasks):
            rejectable += costs[job]

    return rejected, rejected + rejectable, rejected


@pytest.mark.env
@pytest.mark.parametrize("transition", ["des", "tetris"])
def test_rejection_cost_matches_a_computation_from_scratch(
    transition: str,
) -> None:
    rng = Random(13)

    for _ in range(25):
        costs = [float(rng.randint(0, 9)) for _ in range(6)]
        objective = RejectionCost(rejection_costs=costs)
        env = _incremental_env(objective, transition, rng)
        env.reset()

        checkpoints: list[int] = []
        for _ in range(60):
            if env.state.infeasible:
                break

            assert (objective.lb, objective.ub, objective.value) == (
                pytest.approx(_reference_rejection(costs, env.state))
            )

            if env.state.is_terminal():
                break

            choice = rng.random()
            support = env.get_action_support()

            if choice < 0.15:
                checkpoints.append(env.checkpoint())

            elif choice < 0.25 and checkpoints:
                env.backtrack(checkpoints.pop())

            elif choice < 0.5 and support:
                env.step(("reject", rng.choice(support)))

            else:
                env.step(("execute", rng.choice(support)) if support else None)


def test_negative_rejection_costs_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SchedulingEnv(
            SingleMachineSetup(),
            [OptionalityConstraint()],
            RejectionCost(rejection_costs=[-1.0]),
            instance={"processing_time": [1]},
        )

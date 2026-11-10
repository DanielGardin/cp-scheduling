from typing import Literal

import pytest

from cpscheduler.environment import (
    IdenticalParallelMachineSetup,
    Makespan,
    ReleaseDateConstraint,
    SchedulingEnv,
    SingleMachineSetup,
    TotalTardiness,
    WeightedCompletionTime,
    WeightedTardiness,
)
from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.heuristics.pdrs import (
    FirstInFirstOut,
    ModifiedDueDate,
    ShortestProcessingTime,
    WeightedModifiedDueDate,
    WeightedShortestProcessingTime,
)
from cpscheduler.heuristics.pdrs.base import select_task


def _weighted_observation(
    processing: list[int], weights: list[float]
) -> DefaultObservation:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=WeightedCompletionTime(),
        observation=DefaultObservation(),
        instance={"processing_time": processing, "weight": weights},
    )
    obs, _ = env.reset()

    return obs


def test_wspt_ranks_by_weight_over_processing_time() -> None:
    obs = _weighted_observation([1, 4, 2], [10.0, 1.0, 1.0])
    rule = WeightedShortestProcessingTime()

    assert rule.priority_score(obs) == [10.0, 0.25, 0.5]
    assert rule(obs) == ("execute", 0)


def test_wspt_ranks_instantaneous_tasks_first() -> None:
    obs = _weighted_observation([0, 1], [1.0, 1.0])

    scores = WeightedShortestProcessingTime().priority_score(obs)

    assert scores[0] > scores[1]


def test_fifo_selects_the_earliest_release() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [ReleaseDateConstraint()],
        Makespan(),
        observation=DefaultObservation(),
        instance={"processing_time": [1, 4], "release_time": [1, 0]},
    )
    obs, _ = env.reset()

    assert FirstInFirstOut()(obs) == ("execute", 1)


def test_modified_due_date_uses_the_current_time() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=TotalTardiness(),
        observation=DefaultObservation(),
        instance={"processing_time": [5, 5, 1], "due_date": [1, 8, 9]},
    )
    obs, _ = env.reset()
    rule = ModifiedDueDate()

    assert rule.priority_score(obs) == [-5, -8, -9]

    obs, *_ = env.step(("execute", 0))

    # At t = 5, max(t + p_j, d_j) is 10, 10 and 9
    assert env.backend.get_info()["current_time"] == 5
    assert rule.priority_score(obs) == [-10, -10, -9]
    assert rule(obs) == ("execute", 2)


def test_rules_have_a_task_on_the_first_decision() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        [ReleaseDateConstraint()],
        Makespan(),
        observation=DefaultObservation(),
        instance={"processing_time": [1, 4], "release_time": [5, 5]},
    )
    obs, _ = env.reset()

    assert ShortestProcessingTime()(obs) == ("execute", 0)


def test_skipping_an_empty_selection_keeps_the_episode_running() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        instance={"processing_time": [1]},
    )
    env.reset()

    *_, terminated, truncated, _ = env.step(
        select_task([], [], resolve_empty="skip")
    )

    assert (terminated, truncated) == (False, False)
    assert env.state.remaining_tasks == 1


@pytest.mark.parametrize("scheme", ["serial", "parallel"])
def test_ranking_mid_episode_leaves_out_scheduled_tasks(
    scheme: Literal["serial", "parallel"],
) -> None:
    env = SchedulingEnv(
        IdenticalParallelMachineSetup(1),
        objective=Makespan(),
        observation=DefaultObservation(),
        instance={"processing_time": [3, 1, 2]},
    )
    env.reset()
    obs, *_ = env.step(("execute", 0))

    ranking = ShortestProcessingTime().ranking(obs, scheme)

    instruction = "execute" if scheme == "serial" else "submit"
    assert ranking == [(instruction, 1), (instruction, 2)]

    *_, terminated, truncated, _ = env.step(ranking)

    assert (terminated, truncated) == (True, False)


def test_weighted_modified_due_date_follows_kanet_and_li() -> None:
    # At t = 10, max(p, d - t) / w is 1 for task 1 and 2.5 for task 2
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=WeightedTardiness(),
        observation=DefaultObservation(),
        instance={
            "processing_time": [10, 1, 5],
            "due_date": [100, 0, 0],
            "weight": [1.0, 1.0, 2.0],
        },
    )
    env.reset()
    obs, *_ = env.step(("execute", 0))

    assert env.backend.get_info()["current_time"] == 10
    assert WeightedModifiedDueDate()(obs) == ("execute", 1)

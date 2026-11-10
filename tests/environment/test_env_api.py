import pytest

from cpscheduler.environment import (
    ComposedObjective,
    HorizonConstraint,
    Makespan,
    Objective,
    PrecedenceConstraint,
    ReleaseDateConstraint,
    SchedulingEnv,
    SetupConstraint,
    SingleMachineSetup,
    TotalCompletionTime,
)
from cpscheduler.environment.objectives import SatisfactionObjective, Sense
from cpscheduler.environment.reward import (
    LBPotentialRewardStrategy,
    RewardStrategy,
    SparseRewardStrategy,
)


def test_repr_works_before_loading_an_instance() -> None:
    env = SchedulingEnv(
        machine_setup=SingleMachineSetup(),
        constraints=[PrecedenceConstraint({1: [0]})],
        objective=Makespan(),
    )

    assert "chains" in repr(env)
    assert env.get_entry() in repr(env)


def test_graham_entry_has_no_empty_terms() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),  # 1
        [
            ReleaseDateConstraint(),  # r_j
            SetupConstraint(),  # s_jk
            HorizonConstraint(10),  # %empty%
        ],
        Makespan(),  # C_max
    )

    assert env.get_entry() == "1|r_j,s_jk|C_max"


def test_string_is_not_an_action() -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        instance={"processing_time": [1]},
    )
    env.reset()

    # A string is an iterable, it must not be read as a list of instructions
    with pytest.raises(ValueError, match="instruction tuple"):
        env.step("execute")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("objective", "sense", "expected"),
    [
        pytest.param(Makespan(), None, "minimize", id="objective_default"),
        pytest.param(None, None, "maximize", id="satisfaction_default"),
        pytest.param(Makespan(), "maximize", "maximize", id="explicit"),
        pytest.param(
            # A composition ignores the defaults of its terms
            ComposedObjective([SatisfactionObjective(), TotalCompletionTime()]),
            None,
            "minimize",
            id="composition",
        ),
    ],
)
def test_the_environment_decides_the_sense(
    objective: Objective | None, sense: Sense | None, expected: Sense
) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=objective,
        sense=sense,
        instance={"processing_time": [2]},
    )

    assert env.sense == expected
    assert env.minimize == (expected == "minimize")


def test_unknown_sense_is_rejected() -> None:
    # Literal values are not checked by the compiled code
    with pytest.raises(ValueError, match="sense"):
        SchedulingEnv(
            SingleMachineSetup(),
            objective=Makespan(),
            sense="min",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("reward", "magnitude"),
    [
        # Sparse returns the final makespan, 5. The LB potential returns its
        # change since the start, when the bound is the longest task, 3.
        pytest.param(SparseRewardStrategy(), 5, id="sparse"),
        pytest.param(LBPotentialRewardStrategy(), 5 - 3, id="lb_potential"),
    ],
)
@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_the_sense_signs_the_reward(
    sense: Sense, reward: RewardStrategy, magnitude: float
) -> None:
    env = SchedulingEnv(
        SingleMachineSetup(),
        objective=Makespan(),
        reward=reward,
        sense=sense,
        instance={"processing_time": [2, 3]},
    )
    env.reset()

    _, step_return, terminated, _, _ = env.step(
        [("execute", 0), ("execute", 1)]
    )

    assert terminated
    assert step_return == (-magnitude if sense == "minimize" else magnitude)

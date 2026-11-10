from collections.abc import Callable, Sequence

import pytest

from cpscheduler.environment.constants import MAX_TIME
from cpscheduler.environment.instance import Feature, ProblemInstance
from cpscheduler.environment.instance.instance import InstanceStatus
from cpscheduler.environment.setups import IdenticalParallelMachineSetup

InstanceData = dict[str, list[int]]


def _task_feature(
    name: str,
    *,
    optional: bool = False,
    owner: bool = False,
    preprocess: Callable[[Sequence[int]], list[int]] | None = None,
) -> Feature[list[int]]:
    if preprocess is None:
        return Feature(
            name,
            shape=("n_tasks",),
            value_type="time",
            optional=optional,
            owner=owner,
        )

    return Feature(
        name,
        shape=("n_tasks",),
        value_type="time",
        optional=optional,
        owner=owner,
        preprocess=preprocess,
    )


def _instance(*features: Feature[list[int]]) -> ProblemInstance:
    instance = ProblemInstance(debug_mode=True)

    for feature in features:
        instance.register(feature)

    return instance


def _load(
    instance: ProblemInstance, *data: InstanceData, n_machines: int = 2
) -> ProblemInstance:
    instance.initialize(data, IdenticalParallelMachineSetup(n_machines))

    return instance


@pytest.fixture
def loaded() -> ProblemInstance:
    return _load(_instance(_task_feature("due_date")), {"due_date": [0, 0]})


def test_only_features_without_a_provider_are_required() -> None:
    due = _task_feature("due_date")
    weight = _task_feature("weight", optional=True)
    release = _task_feature("release_date")
    instance = _instance(
        due, weight, release, _task_feature("release_date", owner=True)
    )

    assert instance.required_features(show_optional=False) == {
        "due_date": due.metadata
    }
    assert "weight" in instance.required_features(show_optional=True)
    assert "release_date" not in instance.required_features(show_optional=True)


@pytest.mark.parametrize("name", ["all_processing_times", "machine_mask"])
def test_reserved_feature_names_are_rejected(name: str) -> None:
    instance = ProblemInstance(debug_mode=True)

    with pytest.raises(ValueError, match="reserved"):
        instance.register(Feature(name))


def test_registration_is_rejected_once_loaded(
    loaded: ProblemInstance,
) -> None:
    with pytest.raises(RuntimeError):
        loaded.register(_task_feature("weight"))


def test_every_task_is_its_own_job_without_job_data(
    loaded: ProblemInstance,
) -> None:
    assert loaded.job_ids == [0, 1]
    assert loaded.job_tasks == [[0], [1]]


def test_tasks_are_grouped_by_job_in_task_order() -> None:
    instance = _load(
        _instance(_task_feature("due_date")),
        {"due_date": [1, 2, 3, 4], "job": [1, 0, 1, 0]},
    )

    assert instance.n_jobs == 2
    assert instance.job_tasks == [[1, 3], [0, 2]]


def test_tasks_start_without_eligible_machines(
    loaded: ProblemInstance,
) -> None:
    assert loaded.processing_times == [[MAX_TIME] * 2] * 2
    assert all(loaded.get_machines(task) == [] for task in range(2))


def test_later_instances_override_earlier_ones() -> None:
    instance = _load(
        _instance(_task_feature("due_date"), _task_feature("weight")),
        {"due_date": [1, 2], "weight": [1, 1]},
        {"weight": [3, 4]},
    )

    assert instance.get_feature_data("due_date") == [1, 2]
    assert instance.get_feature_data("weight") == [3, 4]
    assert instance.original_instance == {"due_date": [1, 2], "weight": [3, 4]}


def test_loaded_data_is_not_aliased_with_the_input() -> None:
    data = {"due_date": [1, 2]}
    instance = _load(_instance(_task_feature("due_date")), data)

    data["due_date"].append(3)

    assert instance.get_feature_data("due_date") == [1, 2]
    assert instance.n_tasks == 2


def test_every_consumer_of_a_feature_is_loaded() -> None:
    first = _task_feature("due_date")
    second = _task_feature("due_date")

    _load(_instance(first, second), {"due_date": [1, 2]})

    assert first.value == second.value == [1, 2]


def test_preprocessing_is_applied_on_load() -> None:
    feature = _task_feature(
        "due_date", preprocess=lambda data: [2 * x for x in data]
    )

    instance = _load(_instance(feature), {"due_date": [1, 2]})

    assert instance.get_feature_data("due_date") == [2, 4]


def test_unregistered_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="no such feature is registered"):
        _load(ProblemInstance(debug_mode=True), {"unknown": [1]})


def test_conflicting_task_counts_are_rejected() -> None:
    instance = _instance(_task_feature("due_date"), _task_feature("weight"))

    with pytest.raises(ValueError, match="n_tasks"):
        _load(instance, {"due_date": [1, 2], "weight": [1, 2, 3]})


def test_machine_count_must_match_the_setup() -> None:
    feature: Feature[list[int]] = Feature(
        "setup_time", shape=("n_machines",), value_type="duration"
    )

    with pytest.raises(ValueError, match="n_machines"):
        _load(_instance(feature), {"setup_time": [1, 2, 3]}, n_machines=2)


def test_job_count_must_match_the_job_data() -> None:
    feature: Feature[list[int]] = Feature(
        "job_weight", shape=("n_jobs",), value_type="time"
    )
    instance = _instance(_task_feature("due_date"), feature)

    with pytest.raises(ValueError, match="n_jobs"):
        _load(
            instance, {"due_date": [1, 2], "job": [0, 0], "job_weight": [1, 1]}
        )


def test_loading_twice_requires_a_reset(loaded: ProblemInstance) -> None:
    with pytest.raises(RuntimeError, match="reset"):
        _load(loaded, {"due_date": [1]})


def test_processing_time_makes_a_machine_eligible(
    loaded: ProblemInstance,
) -> None:
    loaded.set_processing_time(0, 1, 5)

    assert loaded.processing_times[0] == [MAX_TIME, 5]
    assert loaded.get_machines(0) == [1]
    assert loaded.get_machines(1) == []


def test_instantaneous_tasks_are_eligible(loaded: ProblemInstance) -> None:
    loaded.set_processing_time(0, 0, 0)

    assert loaded.get_machines(0) == [0]


@pytest.mark.parametrize(
    ("time", "message"),
    [(-1, "negative"), (MAX_TIME, "too large")],
)
def test_processing_time_out_of_range_is_rejected(
    loaded: ProblemInstance, time: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        loaded.set_processing_time(0, 0, time)

    assert loaded.get_machines(0) == []


def test_processing_time_on_an_unknown_machine_is_rejected(
    loaded: ProblemInstance,
) -> None:
    with pytest.raises(ValueError, match="machine"):
        loaded.set_processing_time(0, 2, 1)


def test_removing_a_machine_makes_it_ineligible(
    loaded: ProblemInstance,
) -> None:
    loaded.set_processing_time(0, 0, 3)
    loaded.set_processing_time(0, 1, 4)

    loaded.remove_machine(0, 0)

    assert loaded.get_machines(0) == [1]
    assert loaded.processing_times[0] == [MAX_TIME, 4]


def test_max_processing_time_ignores_ineligible_machines(
    loaded: ProblemInstance,
) -> None:
    assert loaded.max_processing_time == 0

    loaded.set_processing_time(0, 0, 3)
    loaded.set_processing_time(1, 1, 7)

    assert loaded.max_processing_time == 7


def test_finalize_requires_a_loaded_instance() -> None:
    with pytest.raises(RuntimeError, match="loaded"):
        ProblemInstance(debug_mode=True).finalize()


def test_finalized_instance_is_frozen(loaded: ProblemInstance) -> None:
    loaded.finalize()

    assert loaded.status is InstanceStatus.RUNNING

    with pytest.raises(RuntimeError, match="frozen"):
        loaded.set_processing_time(0, 0, 1)

    with pytest.raises(RuntimeError, match="frozen"):
        loaded.remove_machine(0, 0)

    with pytest.raises(RuntimeError, match="frozen"):
        loaded.set_optionality(0)

    with pytest.raises(RuntimeError):
        loaded.finalize()


def test_consumers_share_the_data_of_the_provider() -> None:
    consumer = _task_feature("release_date")
    provider = _task_feature("release_date", owner=True)
    instance = _instance(_task_feature("due_date"), consumer, provider)
    _load(instance, {"due_date": [1, 2]})

    provider.own_data([10, 20])
    instance.finalize()

    assert instance.get_feature_data("release_date") == [10, 20]
    assert consumer.value is provider.value


def test_required_feature_without_data_is_rejected() -> None:
    instance = _instance(_task_feature("due_date"), _task_feature("weight"))
    _load(instance, {"due_date": [1]})

    with pytest.raises(ValueError, match="'weight' has no provider"):
        instance.finalize()


def test_feature_with_two_providers_is_rejected() -> None:
    instance = _instance(
        _task_feature("due_date"),
        _task_feature("release_date", owner=True),
        _task_feature("release_date", owner=True),
    )
    _load(instance, {"due_date": [1]})

    with pytest.raises(ValueError, match="provider"):
        instance.finalize()


def test_consumer_with_another_shape_is_rejected() -> None:
    consumer: Feature[list[int]] = Feature(
        "release_date", shape=("n_jobs",), value_type="time"
    )
    provider = _task_feature("release_date", owner=True)
    instance = _instance(_task_feature("due_date"), consumer, provider)
    _load(instance, {"due_date": [1]})
    provider.own_data([0])

    with pytest.raises(ValueError, match="not compatible"):
        instance.finalize()


def test_fingerprint_only_depends_on_the_data() -> None:
    def fingerprint(due_dates: list[int]) -> int:
        instance = _load(
            _instance(_task_feature("due_date")), {"due_date": due_dates}
        )
        instance.finalize()

        return instance.fingerprint

    assert fingerprint([1, 2, 3]) == fingerprint([1, 2, 3])
    assert fingerprint([1, 2, 3]) != fingerprint([1, 2, 4])


def test_reset_keeps_registrations_and_nothing_else() -> None:
    due = _task_feature("due_date")
    instance = _load(_instance(due), {"due_date": [1, 2], "job": [0, 0]})
    instance.set_processing_time(0, 0, 3)
    instance.finalize()

    instance.reset()

    assert instance.status is InstanceStatus.UNLOADED
    assert instance.fingerprint == 0
    assert instance.features == {}
    assert instance.original_instance == {}
    assert instance.job_tasks == []
    assert (instance.n_tasks, instance.n_jobs, instance.n_machines) == (0, 0, 0)
    assert instance.registered_features["due_date"] == [due]
    assert not due.loaded


def test_reloading_after_reset_leaks_nothing() -> None:
    instance = _load(
        _instance(_task_feature("due_date")),
        {"due_date": [1, 2, 3], "job": [0, 0, 1]},
    )
    instance.finalize()
    instance.reset()

    instance.register(_task_feature("weight", optional=True))
    _load(instance, {"due_date": [4]}, n_machines=1)

    assert (instance.n_tasks, instance.n_jobs, instance.n_machines) == (1, 1, 1)
    assert instance.job_tasks == [[0]]
    assert instance.processing_times == [[MAX_TIME]]
    assert instance.original_instance == {"due_date": [4]}


def test_reset_restores_the_data_owned_by_a_provider() -> None:
    provider = _task_feature("release_date", owner=True)
    instance = _instance(_task_feature("due_date"), provider)
    _load(instance, {"due_date": [1]})
    provider.own_data([5])
    instance.finalize()

    # An episode mutates the working copy
    provider.value[0] = 99
    instance.reset()

    assert provider.value == [5]

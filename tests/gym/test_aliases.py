import pytest

gym = pytest.importorskip("gymnasium")

# Importing the package is what registers the environments
import cpscheduler.gym  # noqa: E402, F401


def _registered_ids() -> list[str]:
    return [
        env_id
        for env_id, spec in gym.registry.items()
        if str(spec.entry_point).startswith("cpscheduler.gym")
    ]


def test_aliases_are_registered_on_import() -> None:
    # Otherwise the test below is parametrized over nothing and always passes
    assert "Jobshop-v0" in _registered_ids()


@pytest.mark.parametrize("env_id", _registered_ids())
def test_every_alias_can_be_made(env_id: str) -> None:
    kwargs = {"n_machines": 2} if "Parallel" in env_id else {}

    gym.make(env_id, **kwargs)

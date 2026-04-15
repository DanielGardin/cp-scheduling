import pytest

from importlib.machinery import EXTENSION_SUFFIXES


@pytest.mark.compilation
def test_is_env_compiled() -> None:
    import cpscheduler.environment.env as env

    compiled = any(env.__file__.endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    if not compiled:
        pytest.skip("Environment module is running from Python source, not compiled extension.")

    assert compiled

from importlib.machinery import EXTENSION_SUFFIXES

import pytest


@pytest.mark.compilation
def test_is_env_compiled() -> None:
    import cpscheduler.environment.env as env
    from cpscheduler import __compiled__

    compiled = any(
        env.__file__.endswith(suffix) for suffix in EXTENSION_SUFFIXES
    )
    if not compiled and compiled == __compiled__:
        pytest.skip(
            "Environment module is running from Python source, not compiled extension."
        )

    elif compiled != __compiled__:
        raise AssertionError(
            f"Expected __compiled__ to be {compiled}, but got {__compiled__} instead."
        )

    assert compiled

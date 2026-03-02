from pytest import mark

@mark.compilation
def test_is_env_compiled() -> None:
    import cpscheduler.environment.env as env

    assert env.__file__.endswith(".so") or env.__file__.endswith(".pyd")

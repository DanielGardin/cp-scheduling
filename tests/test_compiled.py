
def test_is_variables_compiled() -> None:
    import cpscheduler.environment.variables as variables

    assert variables.__file__.endswith(".so")


def test_is_objectives_compiled() -> None:
    import cpscheduler.environment.objectives as objectives

    assert objectives.__file__.endswith(".so")


def test_is_constraints_compiled() -> None:
    import cpscheduler.environment.constraints as constraints

    assert constraints.__file__.endswith(".so")


def test_is_env_compiled() -> None:
    import cpscheduler.environment.env as env

    assert env.__file__.endswith(".so")
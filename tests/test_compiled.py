from pytest import mark

@mark.compilation
def test_is_tasks_compiled() -> None:
    import cpscheduler.environment.tasks as tasks

    assert tasks.__file__.endswith(".so")

@mark.compilation
def test_is_instructions_compiled() -> None:
    import cpscheduler.environment.instructions as instructions

    assert instructions.__file__.endswith(".so")

@mark.compilation
def test_is_schedule_setup_compiled() -> None:
    import cpscheduler.environment.schedule_setup as schedule_setup

    assert schedule_setup.__file__.endswith(".so")

@mark.compilation
def test_is_objectives_compiled() -> None:
    import cpscheduler.environment.objectives as objectives

    assert objectives.__file__.endswith(".so")

@mark.compilation
def test_is_constraints_compiled() -> None:
    import cpscheduler.environment.constraints as constraints

    assert constraints.__file__.endswith(".so")

@mark.compilation
def test_is_env_compiled() -> None:
    import cpscheduler.environment.env as env

    assert env.__file__.endswith(".so")
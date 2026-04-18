import pytest

import cpscheduler

from cpscheduler.environment.constraints import constraints
from cpscheduler.environment.objectives import objectives
from cpscheduler.environment.schedule_setup import setups


@pytest.mark.parametrize("setup_name", sorted(setups))
def test_all_setups_importable_from_cpscheduler(setup_name: str) -> None:
	assert hasattr(cpscheduler, setup_name)
	assert getattr(cpscheduler, setup_name) is setups[setup_name]


@pytest.mark.parametrize("constraint_name", sorted(constraints))
def test_all_constraints_importable_from_cpscheduler(
	constraint_name: str,
) -> None:
	assert hasattr(cpscheduler, constraint_name)
	assert getattr(cpscheduler, constraint_name) is constraints[constraint_name]

@pytest.mark.parametrize("constraint_name", sorted(constraints))
def test_all_constraints_has_default(
	constraint_name: str,
) -> None:
	# Constraints should be instantiable without arguments, when they are not
	# obtainable at instantiation time. Instead, the constraints should expose
	# methods to set the relevant parameters at initialization time, when the
	# instance is available.

	assert hasattr(cpscheduler, constraint_name)
	constraint_cls = getattr(cpscheduler, constraint_name)
	constraint = constraint_cls()
	assert constraint is not None


@pytest.mark.parametrize("objective_name", sorted(objectives))
def test_all_objectives_importable_from_cpscheduler(
	objective_name: str,
) -> None:
	assert hasattr(cpscheduler, objective_name)
	assert getattr(cpscheduler, objective_name) is objectives[objective_name]

@pytest.mark.parametrize("objective_name", sorted(objectives))
def test_all_objectives_has_default(
	objective_name: str,
) -> None:
	# Constraints should be instantiable without arguments, when they are not
	# obtainable at instantiation time. Instead, the objectives should expose
	# methods to set the relevant parameters at initialization time, when the
	# instance is available.

	assert hasattr(cpscheduler, objective_name)
	objective_cls = getattr(cpscheduler, objective_name)
	objective = objective_cls()
	assert objective is not None

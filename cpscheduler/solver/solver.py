"""Solver interface for scheduling problems.

The SchedulingSolver class is a high-level wrapper around the specific
problem formulations and backend solvers.
It does not implement any specific modeling or solving logic itself, but rather
provides a consistent interface for building and solving scheduling problems
using different formulations and solvers.

See the Formulation class and its subclasses for specific modeling approaches.
"""

from copy import deepcopy
from typing import Any, Generic, TypeVar

from cpscheduler.common import AnySchedulingEnv, unwrap_env
from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.des import ActionType
from cpscheduler.environment.utils import InstanceTypes, ensure_iterable
from cpscheduler.solver.formulation import Formulation, formulations

ReturnAction = list[tuple[int, str, int, int]]

R = TypeVar("R")


class SchedulingSolver(Generic[R]):
    """Generic solver interface for scheduling problems.

    This class provides a high-level interface for solving scheduling problems
    using a specified formulation.
    It handles the building of the model, warm starting with an initial solution,
    and extracting the solution after solving with a backend solver.
    """

    env: SchedulingEnv
    formulation: Formulation[R]
    config: dict[str, Any]

    _built: bool = False

    def __init__(
        self,
        env: AnySchedulingEnv,
        formulation: Formulation[R] | str,
        *,
        instance: InstanceTypes | None = None,
        **formulation_kwargs: Any,
    ):
        """Initialize the solver with the given environment and formulation.

        Parameters
        ----------
        env : AnySchedulingEnv
            The scheduling environment to solve.
            It must be loaded with an instance and reset before initializing the solver.

        formulation : Formulation[R] | str
            The formulation to use for solving the scheduling problem.
            If a string is provided, it must be a registered formulation.

        instance : InstanceTypes | None, optional
            An optional instance to load into the environment.
            If provided, the environment will be loaded with this instance and reset.

        **formulation_kwargs : Any
            Additional keyword arguments to pass to the formulation constructor
            if a string identifier is used.

        Raises
        ------
        ValueError
            If the environment is not loaded and reset before initializing the solver.

        """
        env = unwrap_env(env)

        if instance is not None:
            env.load_instance(*ensure_iterable(instance))
            env.reset()

        elif not env.running:
            raise ValueError(
                "Environment must be loaded and reset before initializing the solver."
            )

        if isinstance(formulation, str):
            formulation = formulations[formulation](**formulation_kwargs)

        self.formulation = formulation

        self.formulation.initialize_model(env)

        self.env = env
        self._built = False

    def build(self) -> None:
        """Build the model for the scheduling problem."""
        self.formulation.build(self.env)

        self._built = True

    def warm_start(self, action: ActionType) -> None:
        """Warm start the solver with a valid initial schedule.

        Note that the action can lead either to a complete initial schedule, or
        to a partial schedule.


        Parameters
        ----------
        action : ActionType
            An action representing an initial solution to the scheduling problem.
            It must be accepted by the environment's step function and lead to
            a feasible state.

        Raises
        ------
        ValueError
            If the model has already been built, or if the provided action leads
            to an infeasible state in the environment.

        """
        if self._built:
            raise ValueError(
                "Cannot warm start after the model has been fully built. "
                "Call warm_start before build to set initial variable values"
            )

        cpy_env = deepcopy(self.env)

        cpy_env.step(action)

        if cpy_env.state.infeasible:
            raise ValueError(
                "The provided actions lead to an infeasible state. "
                "Cannot warm start with an infeasible solution."
            )

        self.formulation.warm_start(cpy_env)

    def solve(self, *args: Any, **kwargs: Any) -> tuple[ReturnAction, float, R]:
        """Solve the scheduling problem using the built model."""
        if not self._built:
            self.build()

        result = self.formulation.solve(*args, **kwargs)

        objective_value = self.formulation.get_objective_value()

        actions: list[tuple[int, str, int, int]] = []
        for task_id in self.env.state.runtime.awaiting_tasks:
            machine_id, start_time = self.formulation.get_assignment(task_id)
            actions.append((start_time, "execute", task_id, machine_id))

        actions.sort(key=lambda x: (x[0], x[2]))

        return actions, objective_value, result

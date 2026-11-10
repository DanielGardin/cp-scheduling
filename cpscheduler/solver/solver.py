"""Solver interface for scheduling problems.

The SchedulingSolver class is a high-level wrapper around the specific
problem formulations and backend solvers.
It does not implement any specific modeling or solving logic itself, but rather
provides a consistent interface for building and solving scheduling problems
using different formulations and solvers.

See the Formulation class and its subclasses for specific modeling approaches.
"""

import logging
from copy import deepcopy
from typing import Any, Generic, TypeVar

from cpscheduler.common import AnySchedulingEnv, unwrap_env
from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.backend import ActionType
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.utils import InstanceTypes, ensure_iterable
from cpscheduler.solver.formulation import Formulation, formulations

ReturnAction = list[tuple[Any, ...]]

logger = logging.getLogger(__name__)

R = TypeVar("R")


# FUTURE: It would be interesting to allow the Solver itself to be an
# environment, where the step can commit decisions that interact with the
# internal formulation.
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
    _warm_env: SchedulingEnv | None = None

    found_solution: bool = False
    "Whether the last solve found its own schedule, instead of the warm start."

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
        core_env = unwrap_env(env)

        if instance is not None:
            core_env.load_instance(*ensure_iterable(instance))
            core_env.reset()

        elif not core_env.running:
            raise ValueError(
                "Environment must be loaded and reset before initializing the solver."
            )

        if isinstance(formulation, str):
            formulation = formulations[formulation](**formulation_kwargs)

        self.formulation = formulation

        self.formulation.initialize_model(core_env)

        self.env = core_env
        self._built = False

    def build(self) -> None:
        """Build the model for the scheduling problem."""
        self.formulation.build(self.env)

        # Variables created while building, such as the order of each pair of
        # tasks, only receive their initial values now.
        if self._warm_env is not None:
            self.formulation.warm_start(self._warm_env)

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

        # Before building, the bounds of the schedule tighten the constraints
        self.formulation.warm_start(cpy_env)
        self._warm_env = cpy_env

    def solve(self, *args: Any, **kwargs: Any) -> tuple[ReturnAction, float, R]:
        """Solve the scheduling problem using the built model.

        When the formulation stops without a schedule, the complete warm start
        is returned instead, and `found_solution` is set to False.

        Returns
        -------
        actions : ReturnAction
            Instructions that replay the solution in the environment.
            Backends without a clock receive the tasks ordered by start time,
            all others receive timed instructions.

        objective_value : float
            The objective value of the returned schedule.

        result : R
            The formulation-specific results.

        Raises
        ------
        RuntimeError
            If the formulation found no schedule and there is no complete warm
            start to fall back to.

        """
        if not self._built:
            self.build()

        form = self.formulation
        state = self.env.state
        tasks = [
            task_id
            for task_id in range(state.n_tasks)
            if not state.is_fixed(task_id)
        ]

        result = form.solve(*args, **kwargs)
        self.found_solution = form.has_solution()

        if self.found_solution:
            rejected = [
                task_id for task_id in tasks if not form.is_present(task_id)
            ]
            schedule = [
                (
                    form.get_assignment(task_id)[1],
                    task_id,
                    form.get_assignment(task_id)[0],
                )
                for task_id in tasks
                if task_id not in rejected
            ]

            return (
                self._actions(schedule, rejected),
                form.get_objective_value(),
                result,
            )

        warm_env = self._warm_env
        if warm_env is None or not warm_env.state.is_terminal():
            raise RuntimeError(
                f"{type(form).__name__} did not find a solution, and there is "
                "no complete warm start to return instead."
            )

        logger.warning(
            "%s did not find a solution, returning the warm start.",
            type(form).__name__,
        )

        warm_state = warm_env.state
        schedule = [
            (
                warm_state.get_start(task_id),
                task_id,
                warm_state.get_assignment(task_id),
            )
            for task_id in tasks
            if warm_state.is_assigned(task_id)
        ]
        rejected = [
            task_id for task_id in tasks if warm_state.is_absent(task_id)
        ]

        return (
            self._actions(schedule, rejected),
            warm_env.objective.value,
            result,
        )

    def _actions(
        self,
        schedule: list[tuple[Time, TaskID, MachineID]],
        rejected: list[TaskID],
    ) -> ReturnAction:
        """Return the instructions that replay a (start, task, machine) schedule.

        Rejected tasks come first, so no executed task waits for them.
        """
        schedule.sort()
        actions: ReturnAction = [("reject", task_id) for task_id in rejected]

        if self.env.backend.accepts_timed_instructions:
            actions.extend(
                (start_time, "execute", task_id, machine_id)
                for start_time, task_id, machine_id in schedule
            )

        else:
            actions.extend(
                ("execute", task_id, machine_id)
                for _, task_id, machine_id in schedule
            )

        return actions

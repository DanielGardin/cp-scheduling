from typing import Any
from collections.abc import Sequence
from typing_extensions import NamedTuple

from copy import deepcopy

from cpscheduler.environment.des import SingleInstruction

from cpscheduler.environment import SchedulingEnv
from cpscheduler.common import unwrap_env

from cpscheduler.solver.formulation import Formulation, formulations


class SolverResult(NamedTuple):
    action: Sequence[SingleInstruction]
    objective_value: float
    status: str


class SchedulingSolver:
    env: SchedulingEnv
    formulation: Formulation
    config: dict[str, Any]

    _built: bool = False

    def __init__(
        self,
        env: Any | SchedulingEnv,
        formulation: Formulation | str,
        **formulation_kwargs: Any,
    ):
        self.env = unwrap_env(env)

        if isinstance(formulation, str):
            formulation = formulations[formulation](**formulation_kwargs)

        self.formulation = formulation

        if not self.env.state.loaded:
            raise ValueError(
                "Environment must be loaded before initializing the solver."
            )

        self.formulation.initialize_model(self.env)
        self._built = False

    def build(self) -> None:
        "Build the model for the scheduling problem."
        self.formulation.build(self.env)

        self._built = True

    def warm_start(self, actions: Sequence[SingleInstruction]) -> None:
        if self._built:
            raise ValueError(
                "Cannot warm start after the model has been fully built. "
                "Call warm_start before build to set initial variable values"
            )

        cpy_env = deepcopy(self.env)

        if cpy_env.force_reset:
            cpy_env.reset()

        cpy_env.step(actions)
        self.formulation.warm_start(cpy_env)

    def solve(self, *args: Any, **kwargs: Any) -> SolverResult:
        "Solve the scheduling problem using the built model."
        if not self._built:
            self.build()

        status = self.formulation.solve(*args, **kwargs)

        objective_value = self.formulation.get_objective_value()

        actions: list[tuple[int, str, int, int]] = []
        for task_id in self.env.state.runtime.awaiting_tasks:
            machine_id, start_time = self.formulation.get_assignment(task_id)
            actions.append((start_time, "execute", task_id, machine_id))

        actions.sort(key=lambda x: (x[0], x[2]))

        return SolverResult(
            action=actions, objective_value=objective_value, status=status
        )
